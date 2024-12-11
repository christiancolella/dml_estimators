import time

class ProgressBar:
    def __init__(self, task_name: str, n_items: int, bar_length: int = 60):
        self.task_name = task_name
        self.n_items = n_items
        self.bar_length = bar_length
        
        self.start_time = time.time()
        
    def print(self, n_completed: int):
        if n_completed > self.n_items:
            return
        
        progress = n_completed / self.n_items
        
        header_string = f' {self.task_name}'
        bar_string = f'|{"*" * round(progress * self.bar_length)}{"-" * round((1 - progress) * self.bar_length)}|'
        progress_string = f'({n_completed}/{self.n_items})'
        
        eta = round((time.time() - self.start_time) / n_completed * (self.n_items - n_completed))
        eta_mins = int(eta / 60)
        eta_secs = eta % 60
        
        eta_string = f'ETA: {eta_mins}:{"0" if eta_secs < 10 else ""}{eta_secs}'

        print(f'{header_string} {bar_string} {progress_string} {eta_string} ', end='\r' if n_completed < self.n_items else '\n')
        