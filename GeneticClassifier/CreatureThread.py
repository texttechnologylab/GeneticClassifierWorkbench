import threading

class CreatureThread(threading.Thread):
    
    def __init__(self, creature, lock, threads, sleep_event):
        threading.Thread.__init__(self)
        self._creature = creature
        self._handled = False
        self._lock = lock
        self._threads = threads
        self._sleep_event = sleep_event
        
    def run(self):
        try:
            self._creature.compute()
        finally:            
            self._lock.acquire()
            self._threads.remove(self)
            self._lock.release()
            self._sleep_event.set()
        
    def is_handled(self):
        return self._handled
    
    def set_handled(self):
        self._handled = True