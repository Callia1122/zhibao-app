# -*- coding: utf-8 -*-
"""
智慧植保 · 云端版（Flask）
================================================
一个文件同时提供：前端页面 + 真实病害识别 API
部署到 Render 后，手机在任意网络访问公网链接即可使用。

环境变量（在 Render 控制台 Environment 里配置，不写死在代码中）：
  DASHSCOPE_API_KEY  : 百炼 API Key（sk- 或 sk-ws- 开头）
  DASHSCOPE_BASE_URL : OpenAI 兼容地址
                       - sk- 普通 Key: https://dashscope.aliyuncs.com/compatible-mode/v1
                       - sk-ws- 工作空间 Key: 控制台“API快接入”里显示的专属地址
  VISION_MODEL       : 模型名，默认 qwen3-vl-plus
  PORT               : 平台自动注入，本地默认 5000
"""

import os
import re
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ============ 从环境变量读取配置 ============
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.environ.get("VISION_MODEL", "qwen3-vl-plus")

PROMPT = """你是资深农业植保专家，请识别图片中的作物病害。
只输出一行，格式为「病害名称|置信度」，置信度是 0-100 的整数。
例如：小麦条锈病|92
要求：只输出这一行，不要任何解释、礼貌用语或 Markdown 标记。
若照片不是作物病害或无法判断，只输出：无法识别|0"""


def parse_answer(text):
    """兼容 病害名称|置信度 与 JSON 两种输出"""
    s = str(text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    disease, confidence = "", 0
    if "|" in s:
        parts = s.split("|")
        disease = parts[0].strip()
        try:
            confidence = int(round(float(parts[1].strip())))
        except (ValueError, IndexError):
            confidence = 0
    else:
        import json
        try:
            obj = json.loads(s)
            disease = str(obj.get("disease") or obj.get("病害") or obj.get("name") or "").strip()
            confidence = int(round(float(obj.get("confidence", 0))))
        except (ValueError, TypeError):
            disease = s.split("\n")[0].strip()[:40]
    if not disease or any(k in disease for k in ("无法识别", "未知", "不能判断", "不确定")):
        confidence = 0
    confidence = max(0, min(100, confidence))
    return {"name": disease or "无法识别", "confidence": confidence}


def recognize(base64_image):
    """调用视觉大模型识别病害"""
    if not DASHSCOPE_API_KEY:
        return {"success": False, "error": "服务端未配置 DASHSCOPE_API_KEY 环境变量"}
    if len(base64_image) > 8 * 1024 * 1024:
        return {"success": False, "error": "图片太大，请换一张更小的照片"}
    try:
        from openai import OpenAI
    except ImportError:
        return {"success": False, "error": "服务端缺少 openai 库"}

    try:
        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=BASE_URL, timeout=90.0)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," + base64_image
                    }},
                ],
            }],
            temperature=0.01,
            max_tokens=200,
        )
        ans = parse_answer(resp.choices[0].message.content)
        app.logger.info("识别成功: %s %s", ans["name"], ans["confidence"])
        return {"success": True, **ans}
    except Exception as e:
        ename = type(e).__name__
        emsg = str(e)
        app.logger.error("识别失败 %s: %s", ename, emsg[:300])
        if ename == "AuthenticationError" or "401" in emsg:
            err = "API Key 无效，请检查 DASHSCOPE_API_KEY"
        elif ename == "NotFoundError" or "404" in emsg or "model not exist" in emsg.lower():
            err = "模型名错误，请检查 VISION_MODEL 是否为 qwen3-vl-plus"
        elif "403" in emsg or "unpurchased" in emsg.lower() or "denied" in emsg.lower():
            err = "模型未开通，或 sk-ws- Key 未配专属 DASHSCOPE_BASE_URL"
        elif ename == "RateLimitError" or "429" in emsg:
            err = "额度用完或调用太频繁，请稍后重试"
        else:
            err = "识别服务错误：" + ename
        return {"success": False, "error": err}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/diagnose", methods=["POST"])
def api_diagnose():
    data = request.get_json(silent=True) or {}
    image = re.sub(r"^data:image/[a-zA-Z0-9.+-]+;base64,", "", str(data.get("image") or ""))
    if not image:
        return jsonify({"success": False, "error": "没有收到图片数据"}), 400
    return jsonify(recognize(image))


@app.route("/health")
def health():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
