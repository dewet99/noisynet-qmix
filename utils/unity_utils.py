

def get_worker_id(filename="worker_id.dat"):
        with open(filename, 'a+') as f:
            f.seek(0)
            val = int(f.read() or 0) + 1
            f.seek(0)
            f.truncate()
            f.write(str(val))
            return val
