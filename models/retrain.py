import os
from datetime import datetime,timedelta
from apscheduler.schedulers.background import BackgroundScheduler #this will schedule in the background to retrain calls train_and_save() for stable and risky1
from core.config import RETRAIN_INTERVAL_DAYS

def get_date_range():
    """
    Gets the start and end dates for retraining, and will train on the most recent 2 years of data

    """
    end=datetime.today().strftime('%Y-%m-%d')
    years_to_train=2
    days_to_train=years_to_train*365
    start=(datetime.today()-timedelta(days=(days_to_train))).strftime('%Y-%m-%d')
    return start,end

def retrain_all():
    """
    Retrains all supervised learning models on fresh data, and is called automatically every RETRAIN_INTERVAL_DAYS days by APScheduler.
    Running bots pick up the new .pkl on their next cycle (models.train.ModelHandle).
    """
    from models.train import train_and_save  # imported lazily so the scheduler can start without ML deps loaded
    print(f"Retraining all models at {datetime.now()}...")
    start,end=get_date_range()
    train_and_save("stable",start,end)
    train_and_save("risky1",start,end)
    print("Retraining has been completed. \n")
    return

def first_run_time(interval_days=RETRAIN_INTERVAL_DAYS, now=None):
    """First retrain happens one full interval after boot, never at boot."""
    return (now or datetime.now()) + timedelta(days=interval_days)

def start_scheduler(interval_days=RETRAIN_INTERVAL_DAYS):
    """
    Starts the background scheduler to run retrain_all() every RETRAIN_INTERVAL_DAYS days. Its called from run.py at the system boot.

    BUG FIXED (audit issue M-01): the original code passed next_run_time=None
    to avoid retraining at boot. In APScheduler 3.x that adds the job PAUSED,
    so retraining never ran at all. The first run is now scheduled one full
    interval after boot.
    """
    scheduler=BackgroundScheduler()
    scheduler.add_job(
        retrain_all, trigger='interval', days=interval_days,
        next_run_time=first_run_time(interval_days), id="retrain_all"
    )
    scheduler.start()
    print(f"Retraining scheduler has started and will run retrain every {interval_days} days")
    return scheduler
