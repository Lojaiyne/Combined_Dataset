from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import sleep

IN1 = DigitalOutputDevice(23)
IN2 = DigitalOutputDevice(24)
PWM = PWMOutputDevice(18, frequency=20000)  # 20 kHz = quiet motor

def forward(speed=0.6):
    IN1.on()
    IN2.off()
    PWM.value = speed  # 0.0 to 1.0

def stop():
    PWM.value = 0
    IN1.off()
    IN2.off()

forward(0.5)
sleep(3)
stop()
