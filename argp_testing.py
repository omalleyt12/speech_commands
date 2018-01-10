import argparse

parser = argparse.ArgumentParser(description="Testing ArgParse functionality")

parser.add_argument('--no-noise',dest="no_noise",action="store_false",default=True)

FLAGS, unparsed = parser.parse_known_args()

print(FLAGS.no_noise)	
