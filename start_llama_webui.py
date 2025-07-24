import os
import subprocess
import time
import requests
import docker
import argparse
import sys
import webbrowser

MODEL_NAME = "deepseek-coder-6.7B-instruct.Q4_K_M.gguf"
MODEL_URL = f"https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/{MODEL_NAME}"
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
LEGACY_PATH = os.path.join(MODEL_DIR, "model.gguf")
INFERENCE_TIMEOUT = 60  # seconds

def run_subprocess(cmd, check=True):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        print(result.stderr.strip())
        sys.exit(1)
    return result.stdout.strip()

def wait_for_chat_endpoint(retries=30):
    print("⏳ Waiting for /v1/chat/completions to accept POST requests...")
    headers = {"Content-Type": "application/json", "Authorization": "Bearer fake"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello?"}]
    }
    for _ in range(retries):
        try:
            r = requests.post("http://localhost:11435/v1/chat/completions", headers=headers, json=payload, timeout=4)
            if r.status_code == 200 and "choices" in r.json():
                print("✅ Chat completion endpoint is responding!")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("❌ Chat endpoint did not become ready in time.")
    return False

def wait_for_api(name, url, retries=30, require_models=False):
    print(f"⏳ Waiting for {name} at {url}...")
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                if require_models:
                    data = r.json()
                    if "models" in data and isinstance(data["models"], list) and len(data["models"]) > 0:
                        print(f"✅ {name} is ready! Model: {data['models'][0]['name']}")
                        return True
                else:
                    print(f"✅ {name} is ready!")
                    return True
        except requests.RequestException:
            pass
        time.sleep(1)
    print(f"❌ {name} not responding or no model loaded.")
    return False

def test_llama_chat_completion():
    url = "http://localhost:11435/v1/chat/completions"
    print("🧪 Testing llama.cpp OpenAI-compatible endpoint...")
    headers = {"Content-Type": "application/json", "Authorization": "Bearer fake"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello, who are you?"}]
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=INFERENCE_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if "choices" in data:
            print("✅ Inference succeeded:")
            print(data["choices"][0]["message"]["content"])
        else:
            print("⚠️ 'choices' not found in response:")
            print(data)
    except Exception as e:
        print(f"❌ Failed OpenAI call: {e}")

def test_cursor_compatibility():
    print("🧪 Testing Cursor-style completion call...")
    headers = {"Content-Type": "application/json", "Authorization": "Bearer sk-fakekey"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Generate a Python function that adds two numbers."}]
    }
    try:
        r = requests.post("http://localhost:11435/v1/chat/completions", headers=headers, json=payload, timeout=INFERENCE_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        print("✅ Cursor-style call succeeded:")
        print(data["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"❌ Cursor test failed: {e}")

def ensure_model_exists():
    if os.path.exists(LEGACY_PATH) and not os.path.exists(MODEL_PATH):
        print(f"🛠 Renaming legacy model.gguf to {MODEL_NAME}")
        os.rename(LEGACY_PATH, MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        print(f"❗ Required model file not found at {MODEL_PATH}")
        os.makedirs(MODEL_DIR, exist_ok=True)
        choice = input(f"❓ Do you want to download {MODEL_NAME} (~4GB)? (y/n): ").strip().lower()
        if choice == "y":
            print("⬇️ Downloading model...")
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("✅ Model downloaded.")
        else:
            print("❌ Cannot continue without a GGUF model.")
            sys.exit(1)

def main(cleanup=False):
    ensure_model_exists()

    print("🚀 Starting llama.cpp stack...")
    run_subprocess("docker compose up -d")

    client = docker.from_env()
    for name in ["llama-server"]:
        try:
            client.containers.get(name).restart()
            print(f"🔄 Restarted container: {name}")
        except docker.errors.NotFound:
            print(f"⚠️ Container not found: {name}")

    wait_for_api("llama.cpp API", "http://localhost:11435/v1/models", require_models=True)
    wait_for_chat_endpoint()
    test_llama_chat_completion()
    test_cursor_compatibility()

    if cleanup:
        print("🧹 Cleaning up...")
        run_subprocess("docker compose down")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true", help="Stop containers after test")
    args = parser.parse_args()
    main(cleanup=args.cleanup)
