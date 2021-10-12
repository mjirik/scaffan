import configparser
import io




config = configparser.ConfigParser()
# config["scaffan"] = {
#     "param1": "aeqr",
#     "pqadlkfaj": 'werq'
# }
#

cfgstr = "" + \
"""
[scaffan]
val1=1
Val 2 = 2
val3 = "werqwrqw"
My Value = " asdlkfaf ea"
"""
# cfgstr = "[scaffan]" \
#          "val1=1" \
#          "val2=2" \
#          'val3="qwerqr"'
config.read_string(cfgstr)

with io.StringIO() as ss:
    config.write(ss)
    ss.seek(0)  # rewind
    print(ss.read())
    # logging.warning(ss.read())

print(dict(config))

print({section: dict(config[section]) for section in config.sections()})
dct = dict(config["scaffan"])
print(dct)

for key in dct:
    print(f"{key}={dct[key]}")