import schedule
import time

def send_message():
    print("It's the time!")

# Schedule the message for the target time
target_time = "23:22"
schedule.every().friday.at(target_time).do(send_message)

print(f"Message scheduled for every Friday at {target_time}...")

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(1)