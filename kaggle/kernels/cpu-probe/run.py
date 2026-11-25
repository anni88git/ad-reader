import urllib.request
print("CPU+internet probe:", urllib.request.urlopen("https://pypi.org/simple/rapidfuzz/", timeout=20).status)
