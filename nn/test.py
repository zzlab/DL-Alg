import sys
import time
for i in range(64):
  time.sleep(0.2)
  sys.stdout.write('-')
  sys.stdout.flush()
