from pyabsa import APCCheckpointManager

checkpoint_name = '/Users/petar.ulev/Documents/absa/checkpoints/pyabsa_checkpoints/fast_lsa_t_Crypto_acc_83.62_f1_84.14_state_dict'
sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint=checkpoint_name)

texts = ['Do you think [ASP]Bitcoin[ASP] is bullish, I am not sure, i know what [ASP]ethereum[ASP] is bearish, actually I think [ASP]bitcoin[ASP] is bullish',
         '"for the latest #cryptocurrency #[ASP]bitcoin[ASP] #blockchain trends, reviews, guides and memes follow  @boxminingnews  on twitter! 2 5 12"'
         '"every week day, i publish a short piece with my thoughts on the most interesting things happening in the #ethereumereum ecosystem.  check it out and subscribe  thedailygwei. the daily gwei daily commentary on the [ASP]ethereumereum[ASP] ecosystem. 5 11 46"',
         'savage rejection from [ASP]#bitcoin[ASP]  i\'m sitting sidelined again from any recent positions.   (remember i\'m a trader).  i\'m happy i took a good portion of profit up high and now i can take the afternoon off from charts.  cheers everyone.',
         'wise words, always remember to take profits! don\'t get greedy. [ASP]alitecoinoin[ASP] trading has been wild this week, lock in your gains guys! #crypto [ASP]#bitcoin[ASP] the wolf of all streets @scottmelker  · may 26, 2020 take profit. pay yourself. don’t be a jackass and give it all back. 8 5 60"',
         '[ASP]bitcoin[ASP] is bearish these days',
         '[ASP]bitcoin[ASP] is bullish these days',
         '[ASP]litecoin[ASP] rekt...',
         '[ASP]ethereum[ASP] to the moon',
         '[ASP]Dogecoin[ASP] is the people\'s crypto',
         'Man, I think [ASP]ethereum[ASP] is rekt, but boy, [ASP]ripple[ASP] is bearish!',
         'for the latest #cryptocurrency #[ASP]bitcoin[ASP] trends, reviews, guides and memes follow @boxminingnews on twitter! 2 5 12',
         'every week day, i publish a short piece with my thoughts on the most interesting things happening in the #[ASP]ethereumereum[ASP] ecosystem. 5 11 46'
        ]

for ex in texts:
    print('----------Classifying::: ')
    result = sent_classifier.infer(ex, print_result=False)
