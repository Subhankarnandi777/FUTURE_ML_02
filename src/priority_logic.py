high_priority_keywords = [
    'payment failed',
    'server down',
    'account hacked',
    'unable login'
]

medium_priority_keywords = [
    'slow',
    'delay',
    'bug',
    'issue'
]

def assign_priority(text):

    text = str(text).lower()

    for word in high_priority_keywords:
        if word in text:
            return "High"

    for word in medium_priority_keywords:
        if word in text:
            return "Medium"

    return "Low"