import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
result = reader.readtext('./examples/908.jpg')
print(result)

# , recog_network='iter_10'