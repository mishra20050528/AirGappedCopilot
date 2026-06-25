def get_runbook(cpu, latency, loss):

    if cpu > 80:
        return open("docs/high_cpu.txt").read()

    elif latency > 100:
        return open("docs/latency_issue.txt").read()

    elif loss > 5:
        return open("docs/packet_loss.txt").read()

    return "Network operating normally."