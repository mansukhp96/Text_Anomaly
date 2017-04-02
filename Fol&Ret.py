def fetchTweets():
	f = open('KashmirTwitterData.txt', 'r')
	l = []

	for line in f.readlines():
		try:
			x = line.split('\t')
			_id = x[3].strip()
			_favs = int(x[6].strip())
			_retwts = int(x[7].strip())
			_follwrs = int(x[12].strip())


			l.append({'id': _id, 
				'favourites':_favs, 
				'retweets':_retwts, 
				'followers': _follwrs})

		except:
			continue

	f.close()

	return l

def main():
	l = fetchTweets()
	for x in l :
		print(x)
		print('--------------------------')


if __name__ == '__main__':
	main()