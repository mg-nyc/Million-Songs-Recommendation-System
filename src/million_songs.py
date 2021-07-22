
# %% [markdown]
# # Million Songs Problem Statement
#
# ### Context
# With the advent of technology, societies have become more efficient with their lives. But at the same time, individual human lives have become much more fast paced and distracted by leaving little time to explore artistic pursuits. Also, the technology has made significant advancements in the ability to coexist with art and general entertainment. In fact, it has made it easier for humans with shortage of time to find and consume good content. Therefore, one of the key challenges for the companies is to be able to figure out what kind of content their customers are most likely to consume. Almost every internet based company's revenue relies on the time consumers spend on their platforms. These companies need to be able to figure out what kind of content is needed in order to increase the time spent by customers on their platform and make their experience better.
# Spotify is one such audio content provider who has got a huge market base across the world. It has grown significantly because of its ability to recommend the ‘best’ next song to each and every customer based on the huge preference database they have gathered over time like millions of customers and billions of songs. This is done by using smart recommendation systems that can recommend songs based on the users’ likes/dislikes
#
# ### Problem Statement
#
# Build a recommendation system to propose the top 10 songs for a user based on the likelihood of listening to those songs.
#
# ### Data Dictionary
# The core data is the Taste Profile Subset released by The Echo Nest as part of the Million Song Dataset. There are two files in this dataset. One contains the details about the song id, titles, release, artist name and the year of release. Second file contains the user id, song id and the play count of users.
#
# #### song_data
# 1. song_id - A unique id given to every song
# 2. title - Title of the song
# 3. Release - Name of the released album
# 4. Artist_name - Name of the artist
# 5. year - Year of release
#
# #### count_data
# 1. user _id - A unique id given to the user
# 2. song_id - A unique id given to the song
# 3. play_count - Number of times the song was played
#
# #### Data Source
#  http://millionsongdataset.com/
#  

# %%
from Functions_Million_Songs import *
from Functions_Million_Songs import interactions_matrix     # made from dfu_small


%load_ext autoreload
%autoreload 2
#%reload_ext autoreload
# %%
init_datsets()
from Functions_Million_Songs import dfs, dfu, dfus, dfu_small  #song, user and joint datasets
dfs.info(), dfu.info()
# %%
print(f'\n\n Only a few songs have been listened to a lot by a specific user, so we keep these "outliers": \n')
dfu[['user_id','play_count']].nlargest(10,'play_count'), dfs.count()

print_stats()

#Some Song titles appear in more than one song
print('\n\n Duplicate Titles example: "Intro"')
dfs.loc[dfs.title=='Intro']

print('\n\nSample data for "Kings Of Leon": \n ')
dfus.loc[dfus.artist_name=='Kings Of Leon'].sample(n=5)  

top_n = 10 
show_top_lists(top_n)

# surprise library -- Data Distributions and 
# model evaluation (need more time to finish)
X = surprise_distribution_model_evaluation(dfus, dfu_small)
print(X) 
# print(results)
# %% [markdown]

# MODEL #1 - Collaborative Filtering - Using Cosine Similarity of User_ids

#### dfus dataset(all users & songs data combo)
#### It is possible to use altenate route(not implmented here) of 
#### just using top <n> songs and users 
# %%
init_model_1_CF_matrices()
from Functions_Million_Songs import tmp_pivot_table, sparse_matrix, dense_matrix

# pick user_id and # of song recommendations desired
user_id_to_recommend_songs_to = '01845f57f5c8b3309233e5a4a7145a7d33ad3d52'
num_of_songs_to_recommend = 5

# Show songs user has already listened to
x = pd.DataFrame(dfus.query('user_id == @user_id_to_recommend_songs_to')[['title','artist_name']])
print(f'\n\nSongs {user_id_to_recommend_songs_to} has already listened to:\n {x[["title","artist_name"]]}')

# Recommend songs:
# find ordinal position of user_id index in tmp_pivot_table(user_id,song_id, play_count)
user_id_index = tmp_pivot_table.index.get_loc(
                tmp_pivot_table[tmp_pivot_table.index == user_id_to_recommend_songs_to]
                .iloc[-1].name)
# get cosine similarity based recommendations
recommended_song_id_column_index = (recommendations_cosine_model_1
                                                    (user_id_index, 
                                                     num_of_songs_to_recommend, 
                                                     dense_matrix))

print(f'\n\nTop {num_of_songs_to_recommend} song recommendations for "{user_id_to_recommend_songs_to}"": \n')
       
get_song_titles_from_ids(recommended_song_id_column_index, tmp_pivot_table)

# %% [markdown]
# MODEL #2 - SVD Matrix Factorization
#### Split data into train,validate and test sets on dfus dataset(all users & songs data combo)
#### same user_id may have been randomly split across train,validate and test sets
#### It is possible to use altenate route(not implmented here) of keeping
#### user_id  & song_id of a user either in train,validate and test but not split across
#### these three sets

# %%
from Functions_Million_Songs import im_train, im_val, im_test
from Functions_Million_Songs import interactions_matrix, get_song_titles_from_ids
interactions_matrix = create_interactions_matrix(dfu_small)
X_train, X_test = train_test_split(dfu_small,    test_size=0.2, random_state=42)
X_train, X_val  = train_test_split(X_train,      test_size=0.1, random_state=42)
u_train, s_train, vt_train, u_val, vt_val, u_test, vt_test = perform_svd_test(X_train, X_val, X_test)

# %% [markdown]
### Accuracy Results:
##### For Train set the accurcy dips and then seems to increase as new features are added. The intersection of train & 
##### The Graph is behaving oddly. For Validate & Test sets he accuracy is going down with adding more latent features
##### test sets suggests 450 might be optimal # of latent features to use.
# %%
latent_features_count = 450
u_train_lat, s_train_lat, vt_train_lat =  u_train[:, :latent_features_count], np.diag(s_train[:latent_features_count]), vt_train[:latent_features_count, :]
u_val_lat, vt_val_lat   =  u_val[:,  :latent_features_count],  vt_val[:latent_features_count,:]
u_test_lat, vt_test_lat =  u_test[:, :latent_features_count],  vt_test[:latent_features_count,:]

im_train_preds = np.around(np.dot(np.dot(u_train_lat, s_train_lat), vt_train_lat))
im_val_preds  =  np.around(np.dot(np.dot(u_val_lat,   s_train_lat), vt_val_lat))
im_test_preds =  np.around(np.dot(np.dot(u_test_lat,  s_train_lat), vt_test_lat))

df_im_train_preds = pd.DataFrame(im_train_preds, index=im_train.index, columns=im_train.columns)
df_im_train_preds

# %%
df_im_val_preds = pd.DataFrame(im_val_preds, index=im_val.index, columns=im_val.columns)
df_im_val_preds

# %%
df_im_test_preds = pd.DataFrame(im_test_preds, index=im_test.index, columns=im_test.columns)
df_im_test_preds

# %%
# interactions_matrix = create_interactions_matrix(dfu_small)
#Enter 'user_index' and 'num_recommendations' for the user_id
from Functions_Million_Songs import  get_song_titles
user_index = '8386c39c5185635558657faab8d365ba46cd792d'   #'01845f57f5c8b3309233e5a4a7145a7d33ad3d52'
num_recommendations = 5
recommended_song_ids = recommendations_svd_model_2(user_index, interactions_matrix, df_im_train_preds, num_recommendations)
recommended_song_ids = recommended_song_ids.index
recommended_song_titles = get_song_titles(recommended_song_ids, dfs)
recommended_song_titles
# %% [markdown]
## MODEL #3 - k-Mean Clustering
##### assumption: no. of times a song is played by different users is used instead of 
##### play_count sum (ie. no. of multiple time a user payed a song- sum of all such play counts )
##### beause popularity of song depend on how many different people listened to it
# %%
most_played_songs_users_selected = heatmap_most_played_songs_users_selected(interactions_matrix, num_songs=100, num_users=100)
sparse_ratings = most_played_songs_users_selected.astype('Sparse').sparse.to_coo()
plot_kmean(sparse_ratings, k_min=2, k_max=50, k_step=4)

# unclear elbow, we can see that at k=4 there is a somewhat sharp decrease in inertia
#although not convincing enough so will try k=10
#group users into clusters
predictions = KMeans(n_clusters=10, algorithm='full', random_state=0).fit_predict(sparse_ratings)
results = pd.concat([most_played_songs_users_selected.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
print(results)
heatmap_cluster_groups(results, num_songs=100, num_users=100)
# %% [markdown]
### Predictions 
# %%
cluster_number = 4
n_users = 75
n_songs = 300
cluster_4_predictions = results[results.group == cluster_number].drop(['index', 'group'], axis=1)
cluster_4_predictions = find_most_played_songs(cluster_4_predictions, n_songs)
cluster_4_predictions = find_most_interacted_users(cluster_4_predictions, n_users)
figure = plt.figure(figsize=(20, 4))
sns.heatmap(cluster_4_predictions)
# %%
cluster_4_predictions.fillna('').head()
# %%
# user_id = 36
# user_id_ratings  = cluster_4_predictions.loc[user_id, :]
# user_id_unrated_songs =  user_id_ratings[user_id_ratings.isnull()]
# avg_ratings = pd.concat([user_id_unrated_songs, cluster_4_predictions.mean()], axis=1, join='inner').loc[:,0]
# avg_ratings.sort_values(ascending=False)[:20]

# %%
