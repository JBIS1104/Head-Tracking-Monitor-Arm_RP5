import argparse

from gpiozero import DigitalOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory


"""
HR8833 (DFRobot DRI0040) single DC motor control example on Raspberry Pi.

Typical wiring for Motor A:
- Driver VCC  -> Pi 3V3 (logic)
- Driver GND  -> Pi GND
- Driver VM   -> External motor supply (3.3V to 10V)
- Driver DIR1 -> Pi GPIO pin (default BCM20)
- Driver PWM1 -> Pi GPIO pin (default BCM21)
- Motor leads -> A01 / A02

Notes:
- Do NOT power motor VM from Raspberry Pi 5V pin for larger motors.
- Keep grounds common between Pi and motor supply.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HR8833 DC motor direction control (H-bridge)"
    )
    parser.add_argument("--dir1", type=int, default=20, help="BCM pin for DIR1")
    parser.add_argument("--pwm1", type=int, default=21, help="BCM pin for PWM1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    factory = LGPIOFactory()
    dir1 = DigitalOutputDevice(args.dir1, pin_factory=factory)
    pwm1 = DigitalOutputDevice(args.pwm1, pin_factory=factory)

    def forward() -> None:
        dir1.on()
        pwm1.on()

    def backward() -> None:
        dir1.off()
        pwm1.on()

    def stop() -> None:
        pwm1.off()

    print("HR8833 motor direction control ready")
    print("Commands: f=forward, b=backward, s=stop, q=quit")
    print(f"Pins: DIR1=GPIO{args.dir1}, PWM1=GPIO{args.pwm1}")

    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "f":
                forward()
                print("Forward")
            elif cmd == "b":
                backward()
                print("Backward")
            elif cmd == "s":
                stop()
                print("Stopped")
            elif cmd == "q":
                break
            else:
                print("Unknown command")
    except KeyboardInterrupt:
        pass
    finally:
        stop()
        dir1.close()
        pwm1.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
