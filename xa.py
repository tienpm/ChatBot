import fire

def hello(name="World", country = "USA"):
  return "Hello %s!" % name + " from %s!" % country

if __name__ == '__main__':
  fire.Fire(hello)
