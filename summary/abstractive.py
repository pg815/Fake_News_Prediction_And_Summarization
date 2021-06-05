import numpy as np
import pandas as pd 
import re
import warnings
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from nltk import download
download('stopwords')
download('wordnet')
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import EarlyStopping
from summary.attention  import AttentionLayer
from summary.clean import clean_text
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from matplotlib import pyplot
pd.set_option("display.max_colwidth", -1)
warnings.filterwarnings("ignore")


class SumaryModel:

    def __init__(self):
        self.reviews = pd.read_csv("news_summary.csv",encoding='latin-1')
    
    def pre_process_data(self):
        
        # Check for any nulls values
        self.reviews.isnull().sum()

        # display some reviews
        for i in range(5):
            print("Review: ", i + 1)
            print(self.reviews.headlines[i])
            print('-' * 80)
            print(self.reviews.text[i])
        self.cleaned_headlines = []
        self.cleaned_text = []

        for headlines in self.reviews['headlines']:
            self.cleaned_headlines.append(clean_text(headlines, remove_stopwords=False))
        print("Headlines processed.")

        for text in self.reviews['text']:
            self.cleaned_text.append(clean_text(text))
        print("Texts processed.")

        # let's see some processed reviews
        for i in range(20):
            print("Review: ",i+1)
            print(self.cleaned_headlines[i])
            print('-'*80)
            print(self.cleaned_text[i])
            print()

    def plot_distributions(self):
        
        text_word_count = []
        headlines_word_count = []
        
        for i in self.cleaned_text:
            text_word_count.append(len(i.split()))
        for i in self.cleaned_headlines:
            headlines_word_count.append(len(i.split()))
        
        length_df = pd.DataFrame({'text': text_word_count, 'headlines': headlines_word_count})
        length_df.hist(bins=15)
        plt.show()

        count = 0
        for i in self.cleaned_text:
            if(len(i.split())<=50):
                count += 1
        print(count/len(self.cleaned_text))

    def train_test_split_data(self):

        self.max_headlines_len=14
        self.max_text_len=50
        
        self.cleaned_text = np.array(self.cleaned_text)
        self.cleaned_headlines = np.array(self.cleaned_headlines)
        
        short_text=[]
        short_headlines=[]
        
        for i in range(len(self.cleaned_text)):
            
            if(len(self.cleaned_headlines[i].split())<= self.max_headlines_len and len(self.cleaned_text[i].split())<=self.max_text_len):
                short_text.append(self.cleaned_text[i])
                short_headlines.append(self.cleaned_headlines[i])
        
        df=pd.DataFrame({'text':short_text,'headlines':short_headlines})
        df['headlines'] = df['headlines'].apply(lambda x : 'start '+ x + ' end')

        self.x_tr,self.x_val,self.y_tr,self.y_val=train_test_split(np.array(df['text']),np.array(df['headlines']),test_size=0.1,random_state=0,shuffle=True)
        print(f" type self.x_tr :{type(self.x_tr)}")
        print(f" x_tr[1] : {self.x_tr[1]}")
        df.head()

    def encode_data(self):

        #prepare a tokenizer for reviews on training data
        self.x_tokenizer = Tokenizer()
        self.x_tokenizer.fit_on_texts(list(self.x_tr))

        thresh=4
        cnt=0
        tot_cnt=0
        freq=0
        tot_freq=0

        for key,value in self.x_tokenizer.word_counts.items():
            tot_cnt=tot_cnt+1
            tot_freq=tot_freq+value
            if(value<thresh):
                cnt=cnt+1
                freq=freq+value

        print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
        print("Total Coverage of rare words:",(freq/tot_freq)*100)

        #prepare a tokenizer for reviews on training data
        self.x_tokenizer = Tokenizer(num_words=tot_cnt-cnt)
        self.x_tokenizer.fit_on_texts(list(self.x_tr))

        #convert text sequences into integer sequences
        x_tr_seq    =   self.x_tokenizer.texts_to_sequences(self.x_tr)
        x_val_seq   =   self.x_tokenizer.texts_to_sequences(self.x_val)

        #padding zero upto maximum length
        self.x_tr    =   pad_sequences(x_tr_seq,  maxlen=self.max_text_len, padding='post')
        self.x_val   =   pad_sequences(x_val_seq, maxlen=self.max_text_len, padding='post')

        #size of vocabulary ( +1 for padding token)
        self.x_voc   =  self.x_tokenizer.num_words + 1
        print(self.x_voc)

        #prepare a tokenizer for reviews on training data
        self.y_tokenizer = Tokenizer()
        self.y_tokenizer.fit_on_texts(list(self.y_tr))

        thresh=6
        cnt=0
        tot_cnt=0
        freq=0
        tot_freq=0

        for key,value in self.y_tokenizer.word_counts.items():
            tot_cnt=tot_cnt+1
            tot_freq=tot_freq+value
            if(value<thresh):
                cnt=cnt+1
                freq=freq+value

        print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
        print("Total Coverage of rare words:",(freq/tot_freq)*100)

        #prepare a tokenizer for reviews on training data
        self.y_tokenizer = Tokenizer(num_words=tot_cnt-cnt)
        self.y_tokenizer.fit_on_texts(list(self.y_tr))

        #convert text sequences into integer sequences
        y_tr_seq    =   self.y_tokenizer.texts_to_sequences(self.y_tr)
        y_val_seq   =   self.y_tokenizer.texts_to_sequences(self.y_val)

        #padding zero upto maximum length
        self.y_tr    =   pad_sequences(y_tr_seq, maxlen=self.max_headlines_len, padding='post')
        self.y_val   =   pad_sequences(y_val_seq, maxlen=self.max_headlines_len, padding='post')

        #size of vocabulary
        self.y_voc  =   self.y_tokenizer.num_words +1

        self.y_tokenizer.word_counts['start'],len(self.y_tr)


        ind=[]
        for i in range(len(self.y_tr)):
            cnt=0
            for j in self.y_tr[i]:
                if j!=0:
                    cnt=cnt+1
            if(cnt==2):
                ind.append(i)

        self.y_tr = np.delete(self.y_tr,ind, axis=0)
        self.x_tr = np.delete(self.x_tr,ind, axis=0)

        ind=[]
        for i in range(len(self.y_val)):
            cnt=0
            for j in self.y_val[i]:
                if j!=0:
                    cnt=cnt+1
            if(cnt==2):
                ind.append(i)

        self.y_val = np.delete(self.y_val,ind, axis=0)
        self.x_val = np.delete(self.x_val,ind, axis=0)

    def encoder_decoder(self):

        K.clear_session()
        self.latent_dim = 320
        embedding_dim=220

        # Encoder
        encoder_inputs = Input(shape=(self.max_text_len,))

        #embedding layer
        enc_emb =  Embedding(self.x_voc, embedding_dim,trainable=True)(encoder_inputs)

        #encoder lstm 1
        encoder_lstm1 = LSTM(self.latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

        #encoder lstm 2
        encoder_lstm2 = LSTM(self.latent_dim,return_sequences=True,return_state=True,dropout=0.2,recurrent_dropout=0.2)
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

        #encoder lstm 3
        encoder_lstm3=LSTM(self.latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
        self.encoder_outputs, self.state_h, self.state_c= encoder_lstm3(encoder_output2)

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))

        #embedding layer
        self.dec_emb_layer = Embedding(self.y_voc, embedding_dim,trainable=True)
        dec_emb = self.dec_emb_layer(decoder_inputs)

        self.decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True,dropout=0.3,recurrent_dropout=0.1)
        decoder_outputs,decoder_fwd_state, decoder_back_state = self.decoder_lstm(dec_emb,initial_state=[self.state_h, self.state_c])

        # Attention layer
        self.attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = self.attn_layer([self.encoder_outputs, decoder_outputs])

        # Concat attention input and decoder LSTM output
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

        #dense layer
        self.decoder_dense =  TimeDistributed(Dense(self.y_voc, activation='softmax'))
        decoder_outputs = self.decoder_dense(decoder_concat_input)

        return encoder_inputs, decoder_inputs,decoder_outputs

    def create_model(self):
        encoder_inputs, decoder_inputs, decoder_outputs = self.encoder_decoder()
        # Define the model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.summary()
        return model

    def train_save_model(self):
        model = self.create_model()
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
        self.history = model.fit([self.x_tr,self.y_tr[:,:-1]], self.y_tr.reshape(self.y_tr.shape[0],self.y_tr.shape[1], 1)[:,1:] ,epochs=1,callbacks=[es],batch_size=128, validation_data=([self.x_val,self.y_val[:,:-1]], self.y_val.reshape(self.y_val.shape[0],self.y_val.shape[1], 1)[:,1:]))
        model.save('model.h5')

    def plot_historyof_model(self):

        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

    def encoder_decoder_attention_layer(self):

        encoder_inputs, decoder_inputs, decoder_outputs = self.encoder_decoder()

        self.reverse_target_word_index=self.y_tokenizer.index_word
        self.reverse_source_word_index=self.x_tokenizer.index_word
        self.target_word_index=self.y_tokenizer.word_index

        # Encode the input sequence to get the feature vector
        self.encoder_model = Model(inputs=encoder_inputs,outputs=[self.encoder_outputs, self.state_h, self.state_c])

        # Decoder setup
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_hidden_state_input = Input(shape=(self.max_text_len,self.latent_dim))

        # Get the embeddings of the decoder sequence
        dec_emb2= self.dec_emb_layer(decoder_inputs)
        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = self.decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

        #attention inference
        attn_out_inf, attn_states_inf = self.attn_layer([decoder_hidden_state_input, decoder_outputs2])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = self.decoder_dense(decoder_inf_concat)

        # Final decoder model
        self.decoder_model = Model(
            [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs2] + [state_h2, state_c2])

    def decode_sequence(self,input_seq):
        # Encode the input as state vectors.
        e_out, e_h, e_c = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1,1))

        # Populate the first word of target sequence with the start word.
        target_seq[0, 0] = self.target_word_index['start']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:

            output_tokens, h, c = self.decoder_model.predict([target_seq] + [e_out, e_h, e_c])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.reverse_target_word_index[sampled_token_index]

            if(sampled_token!='end'):
                decoded_sentence += ' '+sampled_token

            # Exit condition: either hit max length or find stop word.
            if (sampled_token == 'end'  or len(decoded_sentence.split()) >= (self.max_headlines_len-1)):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            e_h, e_c = h, c

        return decoded_sentence

    def seq2summary(self,input_seq):
        newString=''
        for i in input_seq:
            if((i!=0 and i!= self.target_word_index['start']) and i!=self.target_word_index['end']):
                newString=newString+self.reverse_target_word_index[i]+' '
        return newString

    def seq2text(self,input_seq):
        newString=''
        for i in input_seq:
            if(i!=0):
                newString=newString+self.reverse_source_word_index[i]+' '
        return newString

    def get_summary(self):
        # smodel = load_model('model.h5',custom_objects={'AttentionLayer': AttentionLayer()})
        # print(self.decode_sequence(self.x_tr))
        print(self.seq2text(self.x_tr[10]))
        # for i in range(len(self.y_tr[:,:-1])):
        print(self.seq2summary(self.y_tr[:,:-1][10]))
        # result = smodel.predict([self.x_tr,self.y_tr[:,:-1]])
        # print(result.shape)
        # print(self.seq2summary(self.y_tr[1]))
        # print(f"type self.x_tr : {self.x_tr}")
        # print(f"shape self.x_tr : {self.x_tr.shape}")
        # print(f"type self.y_tr[:,:-1] : {self.y_tr[:,:-1]}")
        # print(f"shape self.y_tr[:,:-1] : {self.y_tr[:,:-1].shape}")
        # print()

    def find_summary(self):
        statement = ''' Filmmaker Rohit Shetty has said he's happy and relieved that the Rajinikanth-Akshay Kumar starrer '2.0' isn't releasing on Diwali, the same day as his film 'Golmaal Again'. He added, ""We know if we release a film with another big film, business does get affected."" Earlier, Aamir's 'Secret Superstar', 'Golmaal Again' and '2.0' were scheduled to release on Diwali this year.", "Mumbai, May 11 (PTI) Director Rohit Shetty says he is happy that his upcoming film ""Golmaal 4"" will not have to face competition from Ranjinikanths ""2.0"" at the box office as now both the movies are releasing on separate dates. Earlier Rohits film starring Ajay Devgn, Tabu, Parineeti Chopra and Arshad Warsi, was set to release this Diwali alongside ""2.0"". ""We tried to do that (referring to pushing ahead the release date) but we were not getting the right date. If we come on solo week or normal week then it is ok. But when you clash (at the box office) obviously the window is not that big as far as business is concerned '''
        cleaned_statement = []
        text_word_count = []
        short_text = []
        cleaned_statement.append(clean_text(statement))
        print(cleaned_statement)
        cleaned_text = np.array(cleaned_statement)
        # for i in range(len(cleaned_text)):
        #     if len(cleaned_text[i].split()) <= 50:
        #         short_text.append(cleaned_text[i])
        # print(f"short text : {short_text}")
        df = pd.DataFrame({'text': cleaned_text})
        print(df)
        x_tr = np.array(df['text'])
        print(x_tr)

        x_tokenizer = Tokenizer()
        x_tokenizer.fit_on_texts(list(x_tr))

        thresh = 4
        cnt = 0
        tot_cnt = 0
        freq = 0
        tot_freq = 0

        for key, value in x_tokenizer.word_counts.items():
            tot_cnt = tot_cnt + 1
            tot_freq = tot_freq + value
            if (value < thresh):
                cnt = cnt + 1
            freq = freq + value

        x_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
        x_tokenizer.fit_on_texts(list(x_tr))
        x_tr_seq = x_tokenizer.texts_to_sequences(x_tr)
        x_tr = pad_sequences(x_tr_seq, maxlen=50, padding='post')
        # x_tr = np.delete(x_tr, axis=0)

        # print(x_tr)
        smodel = load_model('model.h5', custom_objects={'AttentionLayer': AttentionLayer()})
        result = smodel.predict([x_tr, x_tr])
        print(result[:1])

    def predict_summary(self,news):
        data = news
        content = {'content-type': "application/x-www-form-urlencoded",'cache-control': "no-cache",}
        model = load_model('model.h5',data,content)
        summary = model.text
        return summary



if __name__ == '__main__':
    model = SumaryModel()
    # model.pre_process_data()
    # model.plot_distributions()
    # model.train_test_split_data()
    # model.encode_data()
    # model.create_model()
    # model.train_save_model()
    # model.plot_historyof_model()
    # model.encoder_decoder_attention_layer()
    # # model.get_summary()
    # # model.find_summary()
    # for i in range(0,1):
    #     print("Review:",model.seq2text(model.x_tr[i]))
    #     print("Original summary:",model.seq2summary(model.y_tr[i]))
    #     print("Predicted summary:",model.decode_sequence(model.x_tr[i].reshape(1,50)))
    #     print("\n")
    data = "Massive rains in pune caused floods in sangvi and many houses get ruined "
    print(model.predict_summary(data))









