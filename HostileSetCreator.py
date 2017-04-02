f = open('Words.txt','r')
string = ""
hostile=[]
word= "Hostile"
while 1:
	line = f.readline()
	if not line:break
    	if word in line:
		hostile.append(line.partition(' ')[0])
print hostile
f.close()

