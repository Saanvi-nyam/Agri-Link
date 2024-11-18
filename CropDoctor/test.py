import json
 
# Opening JSON file
f = open(r'C:\Users\saura\OneDrive\Documents\GitHub\agri-link\CropDoctor\test.py')
 
# returns JSON object as
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
for i in data:
    print(i)
 
# Closing file
f.close()