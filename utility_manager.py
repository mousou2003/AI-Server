import subprocess
import sys


class UtilityManager:
    """Utility functions for the application"""
    
    @staticmethod
    def run_subprocess(cmd, check=True, show_output=False, timeout=None):
        if show_output:
            # Show output in real-time for commands like docker compose
            print(f"🔧 Running: {cmd}")
            result = subprocess.run(cmd, shell=True, text=True, encoding='utf-8', errors='ignore', timeout=timeout)
            if check and result.returncode != 0:
                print(f"❌ Command failed: {cmd}")
                sys.exit(1)
            return result
        else:
            # Capture output for other commands
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=timeout)
            if check and result.returncode != 0:
                print(f"❌ Command failed: {cmd}")
                print(result.stderr.strip())
                sys.exit(1)
            return result
