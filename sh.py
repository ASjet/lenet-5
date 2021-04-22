import RPI.GPIO as GPIO
import time
LED = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED,GPIO,OUT)

while True:
    GPIO.output(LED,GPIO.HIGH)
    time.sleep(1)
    print("GPIO.OUT from HIGH to LOW")
    GPIO.output(LED,GPIO.LOW)
    time.sleep(1)
    GPIO.cleanup()
