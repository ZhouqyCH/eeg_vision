def prompter(question, prompt='[Yes/No]? '):
    while True:
        try:
            i = raw_input('%s %s' % (question, prompt))
        except KeyboardInterrupt:
            return False
        if i.lower() in ('yes','y'):
            return True
        elif i.lower() in ('no','n'):
            return False
