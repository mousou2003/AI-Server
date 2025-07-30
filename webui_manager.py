import time
import requests


class WebUIManager:
    def __init__(self):
        # Open WebUI configuration
        self.config = {
            "name": "open-webui",
            "port": 3000,
            "url": "http://localhost:3000"
        }
        
    def wait_for_api(self, retries=30):
        """Wait for WebUI to be ready"""
        print(f"⏳ Waiting for {self.config['name']}...")
        for _ in range(retries):
            try:
                r = requests.get(self.config["url"], timeout=2)
                if r.status_code == 200:
                    print(f"✅ {self.config['name']} is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        print(f"❌ {self.config['name']} not responding.")
        return False
