import pyautogui
import time

print("Testing media keys in 3 seconds. Please open a media player (like YouTube in Chrome or Spotify).")
time.sleep(3)

print("Sending 'playpause' key...")
pyautogui.press('playpause')

print("Done. Did it work?")