import copy
import re
import nltk
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import kstest
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

hostile = ['ABHOR', 'ABOLISH', 'ABRASIVE', 'ABSCOND', 'ABSENTEE', 'ABUSE', 'ABUSE', 'ACCOST', 'ACCURSED', 'ACCUSATION', 'ACCUSE', 'ACCUSE', 'ACRIMONIOUS', 'ACRIMONY', 'ACT', 'ADMONISH', 'ADVERSARY', 'AFFLICT', 'AGGRAVATE', 'AGGRAVATION', 'AGGRESSION', 'AGGRESSIVE', 'AGGRESSIVENESS', 'AGGRESSOR', 'AGGRIEVE', 'AGITATOR', 'ALIENATE', 'ALLEGATION', 'ALTERCATION', 'AMBUSH', 'AMBUSH', 'ANARCHIST', 'ANARCHY', 'ANGER', 'ANGER', 'ANGRY', 'ANGUISH', 'ANIMOSITY', 'ANNIHILATE', 'ANNIHILATION', 'ANNOY', 'ANNOYANCE', 'ANTAGONISM', 'ANTAGONIST', 'ANTAGONISTIC', 'ANTAGONIZE', 'ARGUE', 'ARGUMENT', 'ARM', 'ARMED', 'ARMED', 'ARREST', 'ARREST', 'ARROW', 'ASSAIL', 'ASSAILANT', 'ASSASSIN', 'ASSASSINATE', 'ASSAULT', 'ASSAULT', 'ATTACK', 'ATTACK', 'ATTACK', 'ATTACKER', 'AUSTERE', 'AVENGE', 'AVERSION', 'AVERT', 'AVOID', 'AVOIDANCE', 'BANDIT', 'BANISH', 'BAR', 'BARBARIAN', 'BARBAROUS', 'BASTARD', 'BATTLE', 'BATTLEFIELD', 'BEASTLY', 'BEAT', 'BEAT', 'BEHEAD', 'BELIE', 'BELITTLE', 'BELLIGERENT', 'BELT', 'BEREAVE', 'BERSERK', 'BESIEGE', 'BETRAY', 'BETRAYAL', 'BEWARE', 'BIT', 'BITCHY', 'BITE', 'BITE', 'BITE', 'BITTER', 'BLAME', 'BLAME', 'BLIND', 'BLOCK', 'BLOODSHED', 'BLOODTHIRSTY', 'BLOW', 'BLOW', 'BLURT', 'BOMB', 'BOMB', 'BOMBARD', 'BOMBARDMENT', 'BOTHER', 'BOUT', 'BOX', 'BOYCOTT', 'BOYCOTT', 'BRAND', 'BRANDISH', 'BRAWL', 'BREACH', 'BREAK', 'BREAK', 'BRISTLE', 'BROKE', 'BROKE', 'BRUSQUE', 'BRUTALITY', 'BRUTE', 'BRUTISH', 'BUG', 'BULLET', 'BURGLAR', 'BURGLARY', 'BUTCHERY', 'CALLOUS', 'CANNIBAL', 'CANNON', 'CAPTURE', 'CAST', 'CATACLYSM', 'CAUGHT', 'CENSURE', 'CHALLENGE', 'CHARGE', 'CHARGE', 'CHARGE', 'CHARGE', 'CHASE', 'CHASE', 'CHASTISE', 'CHEAT', 'CHIDE', 'CHIP', 'CHOKE', 'CLASH', 'CLUB', 'CLUB', 'COLD', 'COLD', 'COLLIDE', 'COLLISION', 'COMBAT', 'COMBAT', 'COMBATANT', 'COMPEL', 'COMPETE', 'COMPETITION', 'COMPETITIVE', 'COMPETITOR', 'COMPLAIN', 'COMPLAINT', 'COMPULSION', 'CONCEAL', 'CONCEIT', 'CONDEMN', 'CONDEMN', 'CONDEMNATION', 'CONDESCENDING', 'CONDESCENSION', 'CONFLICT', 'CONFLICT', 'CONFRONT', 'CONFRONTATION', 'CONSPIRACY', 'CONSTRAIN', 'CONTAMINATE', 'CONTAMINATION', 'CONTEMPT', 'CONTEMPTIBLE', 'CONTEMPTUOUS', 'CONTEND', 'CONTEST', 'CONTRADICT', 'CONVICT', 'CORRUPTION', 'COUP', 'CRANKY', 'CRASS', 'CRIMINAL', 'CRIPPLE', 'CRITIC', 'CRITICIZE', 'CROOKED', 'CROP', 'CROSS', 'CROSS', 'CROSS', 'CRUEL', 'CRUELTY', 'CRUSH', 'CRUSHING', 'CURSE', 'CURSE', 'CURT', 'CUT', 'CYNICAL', 'DAGGER', 'DAMAGE', 'DAMN', 'DAMNABLE', 'DAMNED', 'DEADLOCK', 'DEADLY', 'DEAL', 'DECEIT', 'DECEITFUL', 'DECEIVE', 'DECEIVE', 'DECEPTION', 'DECEPTIVE', 'DEFEAT', 'DEFENSIVE', 'DEFIANCE', 'DEFIANT', 'DEFILE', 'DEFY', 'DEGRADE', 'DEMEAN', 'DEMOLISH', 'DEMON', 'DEMORALIZE', 'DENIAL', 'DENOUNCE', 'DENY', 'DEPLORE', 'DEPOSE', 'DEPRIVE', 'DERIDE', 'DERISION', 'DEROGATORY', 'DESERT', 'DESPISE', 'DESPISE', 'DESTROY', 'DESTRUCTION', 'DESTRUCTIVE', 'DETERRENT', 'DETEST', 'DEVASTATE', 'DEVASTATION', 'DEVIL', 'DEVILISH', 'DEVIOUS', 'DIABOLIC', 'DIABOLICAL', 'DISAGREE', 'DISAGREEABLE', 'DISAGREEMENT', 'DISAPPROVE', 'DISAVOW', 'DISAVOWAL', 'DISBELIEF', 'DISCORD', 'DISCORDANT', 'DISCOURAGE', 'DISCREDIT', 'DISCREPANT', 'DISCRIMINATION', 'DISGUST', 'DISGUST', 'DISGUST', 'DISLIKE', 'DISLIKE', 'DISMISS', 'DISMISS', 'DISOBEDIENCE', 'DISOBEDIENT', 'DISPLEASURE', 'DISPOSE', 'DISPUTABLE', 'DISPUTE', 'DISPUTE', 'DISRUPT', 'DISRUPTION', 'DISSATISFIED', 'DISSENT', 'DISSENTION', 'DISTORT', 'DISTRUST', 'DISTURB', 'DISTURBANCE', 'DITCH', 'DIVORCE', 'DIVORCE', 'DIVORCE', 'DOUBLE', 'DOUBLE', 'DRAG', 'DRIVE', 'DUMP', 'EGOTISTICAL', 'ENDANGER', 'ENEMY', 'ENFORCE', 'ENGULF', 'ENRAGE', 'ENTANGLEMENT', 'EPITHET', 'ESCAPE', 'ESCAPE', 'EVEN', 'EXCEPTION', 'EXCLUDE', 'EXCLUSION', 'EXCOMMUNICATION', 'EXECUTE', 'EXECUTE', 'EXECUTION', 'EXILE', 'EXPEL', 'EXPLOIT', 'EXPLOIT', 'FAIL', 'FALLOUT', 'FED', 'FEROCIOUS', 'FEROCITY', 'FEUD', 'FIEND', 'FIERCE', 'FIGHT', 'FIGHT', 'FIGHT', 'FIGHT', 'FIGHTER', 'FINE', 'FINGER', 'FIRE', 'FIRE', 'FIRE', 'FIST', 'FLED', 'FLOOR', 'FOE', 'FOOL', 'FOOL', 'FORCE', 'FORCE', 'FOREBODING', 'FOUGHT', 'FRIGHTEN', 'FROWN', 'FROWN', 'FRUSTRATE', 'FUN', 'FURIOUS', 'FURY', 'GERM', 'GET', 'GET', 'GODDAMN', 'GRAB', 'GRAPPLE', 'GRENADE', 'GRUDGE', 'GRUMBLE', 'GUERRILLA', 'GUN', 'GUNMEN', 'HAMPER', 'HANG', 'HARASS', 'HARASSMENT', 'HARM', 'HARM', 'HARSH', 'HASSLE', 'HATE', 'HATE', 'HATE', 'HATER', 'HATRED', 'HAUNT', 'HEARTLESS', 'HEDGE', 'HEINOUS', 'HIDDEN', 'HIDE', 'HIDE', 'HINDER', 'HINDRANCE', 'HIT', 'HOLD', 'HORRIFY', 'HORROR', 'HOSTILE', 'HOSTILITY', 'HUMILIATE', 'HUNT', 'HUNT', 'HUNTER', 'HURT', 'HUSTLE', 'HUSTLER', 'IMPAIR', 'IMPATIENCE', 'IMPEDE', 'IMPEDIMENT', 'IMPLICATE', 'INDICTMENT', 'INDIGNATION', 'INFECT', 'INFECTION', 'INFILTRATION', 'INFLAME', 'INFRINGEMENT', 'INFURIATE', 'INHIBIT', 'INHIBITION', 'INHUMANE', 'INJUNCTION', 'INJURIOUS', 'INJURY', 'INSOLENCE', 'INSOLENT', 'INTERFERE', 'INTERFERENCE', 'INTERRUPT', 'INTERRUPTION', 'INTRUSION', 'IRK', 'IRON', 'IRON', 'IRRITABLE', 'IRRITATION', 'JAGGED', 'JAIL', 'JEER', 'JEOPARDIZE', 'JERK', 'KEEP', 'KICK', 'KICK', 'KICK', 'KIDNAP', 'KILL', 'KILL', 'KILLER', 'KNIFE', 'KNIFE', 'KNOCK', 'KNOCK', 'LAID', 'LAUGH', 'LAWLESS', 'LAY', 'LEAD', 'LIAR', 'LIE', 'LIE', 'LIMIT', 'LIQUIDATE', 'LIQUIDATION', 'LITIGANT', 'LITIGATION', 'LYING', 'LYING', 'MAD', 'MAKE', 'MALICE', 'MALICIOUS', 'MALIGNANT', 'MANGLE', 'MANSLAUGHTER', 'MAR', 'MARKSMAN', 'MASSACRE', 'MERCILESS', 'MIND', 'MINE', 'MISBEHAVE', 'MISLEAD', 'MISSILE', 'MOB', 'MOLEST', 'MONSTER', 'MONSTROUS', 'MURDER', 'MURDER', 'MURDEROUS', 'MUTTER', 'NAG', 'NASTY', 'NAUGHTY', 'NEEDLE', 'NEGATE', 'NEGLECT', 'NEGLECT', 'NIGGER', 'NIGHTMARE', 'NO', 'NO', 'OBJECT', 'OBLITERATE', 'OBNOXIOUS', 'OBSTINATE', 'OBSTRUCT', 'OFFEND', 'OFFENDER', 'OFFENSIVE', 'OMINOUS', 'OPPONENT', 'OPPOSE', 'OPPOSITION', 'OPPRESS', 'OPPRESSION', 'OSTRACIZE', 'OUST', 'OUTLAW', 'OUTRAGE', 'PAN', 'PARASITE', 'PASS', 'PENALTY', 'PENETRATE', 'PENETRATION', 'PERSECUTE', 'PERSECUTION', 'PICK', 'PINCH', 'PISTOL', 'PLAGUE', 'PLIGHT', 'POINT', 'POISONOUS', 'POLLUTE', 'POSSE', 'POUND', 'PREJUDICE', 'PRETEND', 'PRETENSE', 'PROSECUTION', 'PROTEST', 'PROTEST', 'PROVOCATION', 'PROVOKE', 'PROWL', 'PUNCH', 'PUNISH', 'PUSH', 'PUT', 'QUARREL', 'QUARREL', 'QUARRELSOME', 'QUESTION', 'QUIBBLE', 'RAGE', 'RAID', 'RAISE', 'RAVAGE', 'REACTIVE', 'REBEL', 'REBEL', 'REBELLION', 'REBELLIOUS', 'REBUFF', 'REBUKE', 'REBUT', 'RECALCITRANT', 'REFUSAL', 'REFUSE', 'REJECT', 'REJECTION', 'RENOUNCE', 'RENUNCIATION', 'REPEL', 'REPROACH', 'REPULSE', 'RESENT', 'RESENTFUL', 'RESENTMENT', 'RESIST', 'RESISTANCE', 'RESTRAIN', 'RESTRICT', 'RETALIATE', 'RETARD', 'REVENGE', 'REVOLT', 'REVOLUTION', 'REVOLUTIONARY', 'RID', 'RIDE', 'RIDICULE', 'RIDICULE', 'RIFLE', 'RIP', 'RIVAL', 'RIVALRY', 'ROBBER', 'ROBBERY', 'ROGUE', 'ROOT', 'RUFFIAN', 'RUINOUS', 'RUMPLE', 'RUN', 'RUPTURE', 'RUTHLESS', 'RUTHLESSNESS', 'SABOTAGE', 'SARCASM', 'SARCASTIC', 'SAVAGE', 'SCANDALOUS', 'SCARE', 'SCARED', 'SCOLD', 'SCOLD', 'SCORCH', 'SCORN', 'SCORNFUL', 'SCOUNDREL', 'SCOWL', 'SCREW', 'SCUFFLE', 'SEETHE', 'SEGREGATION', 'SEIZE', 'SERVE', 'SERVICE', 'SEVER', 'SHADOW', 'SHAFT', 'SHATTER', 'SHELL', 'SHELL', 'SHOCK', 'SHOOT', 'SHOOT', 'SHOOT', 'SHOT', 'SHOVE', 'SHRED', 'SHREW', 'SHRUG', 'SHRUG', 'SHUDDER', 'SHUDDER', 'SHUN', 'SHUT', 'SICK', 'SIEGE', 'SINISTER', 'SKIRMISH', 'SLAM', 'SLANDER', 'SLANDERER', 'SLANDEROUS', 'SLAP', 'SLASH', 'SLAUGHTER', 'SLAYER', 'SLEAZY', 'SLIGHT', 'SLIGHT', 'SLY', 'SMACK', 'SMASH', 'SMASH', 'SMEAR', 'SNARL', 'SNATCH', 'SPANK', 'SPEAR', 'SPEAR', 'SPITE', 'SPITEFUL', 'SPLIT', 'SPOIL', 'STAB', 'STALL', 'STAMP', 'STARTLE', 'STEAL', 'STEAL', 'STERN', 'STIFLE', 'STING', 'STOLE', 'STOLEN', 'STONE', 'STOP', 'STORM', 'STORMY', 'STRANGLE', 'STRIFE', 'STRIKE', 'STRIKE', 'STRINGENT', 'STRIP', 'STRUCK', 'STRUGGLE', 'STRUGGLE', 'STRUGGLE', 'STUBBORN', 'STUBBORNLY', 'STUBBORNNESS', 'STUN', 'SUBDUE', 'SUBVERSION', 'SUBVERT', 'SUNDER', 'SUPPRESS', 'SUPPRESSION', 'SUSPECT', 'SUSPECT', 'SUSPICION', 'SUSPICIOUS', 'SWEAR', 'SWORD', 'SWORE', 'TABOO', 'TAINT', 'TAMPER', 'TANTRUM', 'TAUNT', 'TAUNT', 'TAX', 'TEAR', 'TEASE', 'TEMPER', 'TEMPEST', 'TENSE', 'TERRORISM', 'TERRORIZE', 'THEFT', 'THIEF', 'THORNY', 'THRASH', 'THREAT', 'THREATEN', 'THWART', 'TIME', 'TIRE', 'TIRED', 'TIRED', 'TNT', 'TORMENT', 'TOUGH', 'TRAITOR', 'TRAMPLE', 'TRAP', 'TREACHEROUS', 'TREACHERY', 'TREASON', 'TREASONOUS', 'TRICK', 'TRICK', 'TRIGGER', 'TRY', 'TURBULENT', 'TURN', 'TURN', 'ULTIMATUM', 'UNDERMINE', 'UNDID', 'UNDO', 'UNDONE', 'UNFAIR', 'UNJUST', 'UNJUSTIFIED', 'UNLEASH', 'UNRULY', 'UNSAFE', 'UNTRUTH', 'UNWILLING', 'UNWILLINGNESS', 'UPRISING', 'UPSET', 'UPSET', 'USURP', 'VENGEANCE', 'VENOM', 'VENOMOUS', 'VICIOUS', 'VICTIM', 'VIE', 'VILLAIN', 'VIOLATE', 'VIOLATION', 'VIOLENCE', 'VIOLENT', 'VIPER', 'WAGE', 'WAIT', 'WALK', 'WAR', 'WARLIKE', 'WARRIOR', 'WEAPON', 'WEED', 'WENCH', 'WHACK', 'WHEEL', 'WHINE', 'WHIP', 'WHIP', 'WHIP', 'WICKED', 'WICKEDNESS', 'WILY', 'WITCH', 'WITCH', 'WITCHCRAFT', 'WITHHELD', 'WITHHOLD', 'WITHSTAND', 'WORRY', 'WOUND', 'WRATH', 'WRECK', 'WRESTLE', 'WRONG']

def tokenize_only(text):
	# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
	tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
	filtered_tokens = []
	# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filtered_tokens.append(token)
	return filtered_tokens

def fetchTweets():
	f = open('KashmirTwitterData.txt', 'r')
	l = {}

	for line in f.readlines():
		try:
			_id = line.split('\t')[3].strip()
			twt = line.split('\t')[4].strip()
			l[_id]=twt

		except IndexError:
			continue

	f.close()

	return l

def getFields(idl):
	f = open('KashmirTwitterData.txt', 'r')
	for line in f.readlines():
		x = line.split('t')
		

def removeTfIdf():
	arr1=[]	
	for key, value in l.iteritems():	
		arr1.append(value)	
	print(len(l))
	
	temp_sentences = []
	sentences = []

	for i in arr1:
		if len(i) > 20:
			sentences.append(i)

	#define vectorizer parameters
	tfidf_vectorizer = TfidfVectorizer(max_df=0.93, max_features=200000, min_df=0.07, stop_words='english', use_idf=True, tokenizer=tokenize_only, ngram_range=(1,5))
	tfidf_matrix = tfidf_vectorizer.fit_transform(sentences) #fit the vectorizer to articles

	terms = tfidf_vectorizer.get_feature_names()
	#get the distance between the articles
	dist = 1 - cosine_similarity(tfidf_matrix)

	scores = []

	for i in range(len(sentences)):
		scores.append(sum(dist[i], 0.0) / (len(dist[i])))

	arr = copy.deepcopy(scores)
	arr = np.array(arr)
	iqr = np.percentile(arr, 75, interpolation= 'higher') - np.percentile(arr, 25, interpolation= 'lower')

	f = open("tf-idf distribution.txt", "w")

	n = len(scores)

	for i in range(n):
		f.write(str(scores[i]))
		f.write("\n")

	mean_dist = sum(scores)/n
	uppr=mean_dist - (1.5*iqr)
	lwr=mean_dist + (1.5*iqr)

	#For mornal distribution
	outlier = list(filter(lambda x: x < (mean_dist - (1.5*iqr)) or x > (mean_dist + (1.5*iqr)), arr))

	newarr=[]
	for ele in range(len(arr)):
		if arr[ele]<uppr or arr[ele]>lwr:
			newarr.append(ele)
	nwarr=[]
	for ele1 in newarr:
		nwarr.append(arr1[ele1])

	l1={}
	print(len(nwarr))
	for key, value in l.iteritems():	
		if value in nwarr:
			pass;
		else:
			l1[key]=value
	print(len(l1))
	return l1

def hostilityfactor(x):
	count=0;
	for i in x:	
		if i.upper() in hostile: 
			count+=1		
	for i in x:		
		if i.lower() in stop:
			x.remove(i)	
	for i in x:		
		if i.lower() == "":
			x.remove(i)	
	frac = float(count)/float(len(x))
	return frac	
def Hostile():
	hos=[]
	top=0;

	for y in arr1:
		y=y.split(' ')
		red=hostilityfactor(y)
		hos.append(red)
		top+=1;
	top =top * 0.005
	print top
	li=[]
	for i in range(int(top)+1): 
		li.append(hos.index(max(hos)))
		hos[hos.index(max(hos))]=0.0	
		
	print li

		
l = fetchTweets()
l2 = removeTfIdf()
arr1=[]	
for key, value in l.iteritems():	
	arr1.append(value)	
	
stop = set(stopwords.words('english'))
Hostile()


sid = SentimentIntensityAnalyzer()

sentences = []
sentences_string = ""

for i in arr1:
	sentences.append(i)
	

sentiments = [None] * len(sentences)

for i in range(len(sentences)):
	sentiments[i] = sid.polarity_scores(sentences[i])["compound"]

#most negative
print(sentences[ sentiments.index(sorted(sentiments)[0])] , sorted(sentiments)[0])


print "-----------------------------------"

#most positive
print(sentences[ sentiments.index(sorted(sentiments)[len(sentiments)-1])], sorted(sentiments)[len(sentiments)-1])

f = open("sentiment distribution.txt", 'w')

for i in range(len(sentiments)):
	f.write(str(sentiments[i]))
	f.write('\n')































