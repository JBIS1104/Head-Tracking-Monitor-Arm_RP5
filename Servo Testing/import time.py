import time
from gpiozero import PWMOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory

PWM_PIN = 13  # BCM pin with hardware PWM: 12, 13, 18, 19
PWM_FREQUENCY = 50
CONTROL_AS_IF_FREQUENCY = 400
PWM_DUTY_AS_IF_400 = 30.0  # Example: 40% @400Hz becomes 5% @50Hz


def convert_duty_for_frequency(duty_percent: float, from_hz: float, to_hz: float) -> float:
    converted = duty_percent * (to_hz / from_hz)
    return max(0.0, min(100.0, converted))


converted_duty_percent = convert_duty_for_frequency(
    PWM_DUTY_AS_IF_400, CONTROL_AS_IF_FREQUENCY, PWM_FREQUENCY
)
PWM_DUTY = converted_duty_percent / 100.0

factory = LGPIOFactory()  # hardware PWM via liblgpio (no daemon required)
pwm = PWMOutputDevice(PWM_PIN, frequency=PWM_FREQUENCY, pin_factory=factory)

try:
    pwm.value = PWM_DUTY
    print(
        f"GPIO {PWM_PIN} at {PWM_FREQUENCY}Hz | "
        f"as-if duty {PWM_DUTY_AS_IF_400:.1f}% @ {CONTROL_AS_IF_FREQUENCY}Hz -> "
        f"actual duty {converted_duty_percent:.2f}%"
    )
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    pwm.close()