from gpiozero import OutputDevice
from time import sleep

solenoid = OutputDevice(17)  # GPIO17 (physical pin 11)

while True:
    solenoid.on()
    print("ON")
    sleep(0.5)

    solenoid.off()
    print("OFF")
    sleep(0.5)
