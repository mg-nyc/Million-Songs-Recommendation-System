
# %%
import pandas as pd
import numpy as np
import seaborn as sns

import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot, iplot
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

import sys
import warnings
warnings.filterwarnings('ignore')
# %%
# initialize global variables to be used in other .py files
dfs, dfu, dfus, dfu_small = 0, 0, 0, 0
interactions_matrix, tmp_pivot_table, sparse_matrix, dense_matrix = 0, 0, 0, 0
im_train, im_val, im_test = 0, 0, 0
def init_datsets():
    global dfs, dfu, dfus, dfu_small, interactions_matrix
    dfs = (pd.read_csv('song_data.csv') 
             .drop_duplicates(subset='song_id', keep='first')
          ) 
    # user ie. listener data       
    dfu = (pd.read_csv('count_data.csv')
            .drop(columns=['Unnamed: 0'])
            .drop_duplicates(subset=['song_id','user_id'], keep='first')
          )
          
    dfs['song_id']= dfs['song_id'].str.strip()
    dfu['song_id']= dfu['song_id'].str.strip()

    dfus= pd.merge(dfu,dfs, on='song_id', how='left') 

    # SMALLER INERACTION MATRIX- dfu_small of Top <n> most listened songs & 
    # Top users who listened to thos top songs. dfu_small needed for
    # computational speed and cater to sparsity of data

    # Top <n>:10,000 Most Listened song_ids:  to limit interaction matrix size from crashing computer
    dfs_im = pd.DataFrame({'song_id':
                                    dfu['song_id']
                                    .value_counts()[:10_000]
                                    .index})

    # Top <n>:100,000 Most Active Listeners: user_ids
    dfu_im = (pd.DataFrame({'user_id':
                                    dfu[dfu['song_id'].isin(dfs_im['song_id'])]
                                    ['user_id']
                                    .value_counts()[:10_000]
                                    .index}))
    
    # SMALLER INERACTION MATRIX- dfu_small
    dfu_small = dfs_im.merge(dfu, how='inner', on='song_id')
    dfu_small = dfu_im.merge(dfu_small, how='inner', on='user_id')
    dfu_small = pd.merge(dfu_small, dfs, how='left', on='song_id')
    interactions_matrix = create_interactions_matrix(dfu_small)
    # interactions_matrix = dfu_small.pivot(index='user_id', columns='song_id', values='play_count')
    # interactions_matrix.fillna(0, inplace=True)

    # user_song_ratings = interactions_matrix.append(interactions_matrix.sum(), ignore_index=True)
    # user_song_ratings_sorted = user_song_ratings.sort_values(len(user_song_ratings)-1, axis=1, ascending=False)
    #interactions_matrix_play_count = dfus.pivot(index='user_id', columns='song_id', values='play_count')

# initialie Model #1 Collaborative filtering- cosine similatity matrices
def init_model_1_CF_matrices():
    global tmp_pivot_table, sparse_matrix, dense_matrix
    #create pivot able of user_id,song_id, play_count
    tmp_pivot_table = (dfus[dfus['user_id'].notna()]
                    .pivot(index='user_id', columns='song_id', values='play_count')
                    )

    #A non-zero  format: tuple (user_id_index, song_id_index) play_count : (416, 0) 2.0
    #note: sparse matrix has ordinal positions of user_id & song_id from tmp_pivot_table
    sparse_matrix = (tmp_pivot_table.astype('Sparse')
                                    .sparse.to_coo()
                    )

    #fully filled zeros & play_counts matrix like  a real math type matrix
    dense_matrix = sparse_matrix.todense()

#pivot table of user_id -- song_id -- playcount
def create_interactions_matrix(df):
    im = df.pivot(index='user_id', columns='song_id', values='play_count')
    im.fillna(0, inplace=True)
    return im

def print_stats():
    print('\n\n Some Listner and Song Stats:')
    total_users = dfu.user_id.nunique()
    print(f'          Total Users:  {total_users}')
    print(f'        Total Artists:  {dfs.artist_name.nunique()}')
    total_Songs = dfs.song_id.nunique()
    print(f'          Total Songs: {total_Songs}')
    print(f'    Total Song Titles: {dfs.title.nunique()}')
    print(f'Total Albums Released: {dfs.release.nunique()}')
    year = dfs.query('year>0')['year'].min()
    print(f'     Oldest Song Year:   {year}')
    year = dfs.query('year>0')['year'].max()
    print(f'     Newest Song Year:   {year}')
    us_density = (len(dfu)/(total_users * total_Songs)) * 100
    print(f'    User-Song Density: {round(us_density,4)}%')

def plot_distributions(dfu_small):
    data = dfu_small['song_id'].value_counts().sort_index(ascending=False)
    trace = go.Bar(x = data.index,
                text = ['{:.1f} %'.format(val) for val in (data.values / dfu_small.shape[0] * 100)],
                textposition = 'auto',
                textfont = dict(color = '#000000'),
                y = data.values,
                )
    # Create layout
    layout = dict(title = 'Distribution Of {} Play Counts'.format(dfu_small.shape[0]),
                xaxis = dict(title = 'Song Ids'),
                yaxis = dict(title = 'Play Count'))
    # Create plot
    fig = go.Figure(data=[trace], layout=layout)
    fig.update_yaxes(type="log")
    iplot(fig)

def show_top_lists(top_n):
    print(f'\n\nTop {top_n} Most Actie Listeners')
    print(top_n_active_listeners(dfus, top_n))

    print(f'\n\nTop {top_n} Most Listened Artist:')
    print(top_n_listened_artist(dfus, top_n))

    print(f'\n\nTop {top_n} most Played Songs:')
    print(top_n_played_songs(dfus, top_n))

    print(f'\n\nTop {top_n} Songs with most number of listeners:')
    x = top_n_songs_with_most_listeners(dfus, top_n)
    print(f'\n\n{x}\n\n')
    fig=px.bar(x=x['title'].str[:20], y=x['Number_of_Listeners'], 
        labels=dict(x='Songs', y='Number_of_Listeners' ), 
        log_y=True, 
        title=f'Top {top_n} Songs with most number of listeners'
        )
    fig.show()

    play_agg = (dfus.groupby(['song_id'])['song_id']
                    .count()
                    .reset_index(name='count') 
                    .sort_values(['count'], ascending=False) 
                    # .head(5)
            )
    print(f'\n\n{play_agg}\n')
    fig= px.bar(x = play_agg['song_id'], y = play_agg['count'], 
                labels=dict(x='Song', y='# of Listeners'), 
                log_y=True, 
                title=f'Distribution of Listeners by Song'
                )
    fig.show()

# Top n Active Users/Listeners based on Number of times Songs listened
def top_n_active_listeners(df, n):
    x =(
        df.groupby('user_id', sort=False)['play_count']
          .count()
          .astype('int')
          .sort_values(ascending=False)
          .head(n)
          .reset_index()
          .assign(Rank=np.arange(n+1)[1:n+1])
          .set_index('Rank')
    )

    return x.loc[:,['user_id', 'play_count']]
 
# Top n Most Listened Artist
#(includes Listners who listen to a Artist lot and can skew results)
# n= Number of top results desired
def top_n_listened_artist(df, n):
    x = (
            df.groupby('artist_name')['play_count']
              .agg('sum')
              .astype('int')
              .sort_values(ascending=False)
              .head(n)
              .reset_index()
              .assign(Rank=np.arange(n+1)[1:n+1])
              .set_index('Rank')
         )

    return x.loc[:,['artist_name', 'play_count']]

# Top n Most Played Songs 
#(includes Listners who listen to a song a lot and can skew results)
# n= Number of top results desired
def top_n_played_songs(df,n):
    x = (
        df.groupby(['title','artist_name']) 
          .agg({'play_count':'sum'})
          .astype('int')
          .sort_values(by=['play_count'], ascending=False)
          .head(n)
          .reset_index()
          .assign(Rank=np.arange(n+1)[1:n+1])
          .set_index('Rank')
         )

    return x.loc[:,['title', 'artist_name', 'play_count']] 

# Top n Songs with most number of Listners 
# each listner of a song is counted only once
def top_n_songs_with_most_listeners(df,n):
    x = (
            df.groupby(['title','artist_name','song_id']) 
              .agg({'user_id':'count'})
              .astype('int')
              .sort_values(by='user_id', ascending=False)
              .head(n)
              .reset_index()
              .assign(Rank=np.arange(n+1)[1:n+1])
              .set_index('Rank')
              .rename(columns={'user_id': 'Number_of_Listeners'})
         )

    return x.loc[:,['title', 'artist_name', 'Number_of_Listeners', 'song_id']]  

def surprise_distribution_model_evaluation(dfus, dfu_small):
    init_notebook_mode(connected=True)
    plot_distributions(dfu_small)
    n = 20  #top <n> songs list
    data = top_n_songs_with_most_listeners(dfus, n)
    # Number of Unique User Plays
    # data = dfu_small.groupby('song_id')['title'].count().clip(upper=50)

    # Create trace
    trace = go.Histogram(x = data.values,
                        name = 'Songs',
                        xbins = dict(start = 0,
                                    end = 50,
                                    size = 2))
    # Create layout
    layout = go.Layout(title = 'Distribution of Number of Unique User Plays of Songs (Clipped at 100)',
                    xaxis = dict(title = 'Songs'),
                    yaxis = dict(title = 'Count'),
                    bargap = 0.2)

    # Create plot
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)

    reader = Reader(rating_scale=(0,9))
    data = Dataset.load_from_df(dfu_small[['user_id', 'song_id', 'play_count']],reader=reader)

    # # Use the famous SVD algorithm.
    # algo = SVD()

    # # Run 5-fold cross-validation and print results.
    # cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    benchmark = []
    # Iterate over all algorithms
    KNNBasic()
    for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
        # Perform cross validation
        results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
        
        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)

    X = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse') 
    return X   #, results


def similar_users(user_id_index, dense_matrix):
    similarity = []
    for user in np.arange(0, dense_matrix.shape[0]):
        sim = cosine_similarity(dense_matrix[user_id_index], dense_matrix[user])
        similarity.append((user, sim))
        
    similarity.sort(key=lambda x: x[1], reverse=True)
    most_similar_users = [tup[0] for tup in similarity]
    similarity_score = [tup[1] for tup in similarity]
    most_similar_users.remove(user_id_index)
    similarity_score.remove(similarity_score[0])
       
    return most_similar_users  #, similarity_score --> use this if score neede anywhere else ignore score

def get_song_titles_from_ids(recommended_song_id_column_index, tmp_pivot_table):
    recommended_song_titles = [] 
    for song_col in recommended_song_id_column_index:
        song_id = tmp_pivot_table.columns[song_col]
        recommended_song_titles.extend(dfus[dfus['song_id']== song_id]
                                      ['title'].unique())

    return recommended_song_titles

def get_song_titles(recommended_song_id, df):
    recommended_song_titles = [] 
    for song_id in recommended_song_id:
        recommended_song_titles.extend(df.query('song_id== @song_id')
                                      ['title'].unique())

    return recommended_song_titles


def find_most_played_songs(interactions_matrix, num_songs):
    user_song_plays = interactions_matrix.append(interactions_matrix.count(), ignore_index=True)
    user_song_plays_sorted = user_song_plays.sort_values(len(user_song_plays)-1, axis=1, ascending=False)
    user_song_plays_sorted = user_song_plays_sorted.drop(user_song_plays_sorted.tail(1).index)
    most_played_songs = user_song_plays_sorted.iloc[:, :num_songs]
                                                    
    return most_played_songs
def find_most_interacted_users(most_played_songs, num_users):
    most_played_songs['counts'] = pd.Series(most_played_songs.count(axis=1))
    most_played_songs_users = most_played_songs.sort_values('counts', ascending=False)
    most_played_songs_users_selected = most_played_songs_users.iloc[:num_users, :]
    most_played_songs_users_selected = most_played_songs_users_selected.drop(['counts'], axis=1)
    
    return most_played_songs_users_selected

def heatmap_most_played_songs_users_selected(interactions_matrix, num_songs=100, num_users=100):
    most_played_songs = find_most_played_songs(interactions_matrix, num_songs)
    most_played_songs_users_selected = find_most_interacted_users(most_played_songs, num_users)
    plt.figure(figsize=(20, 10), facecolor='gray')
    sns.heatmap(most_played_songs_users_selected)
    return most_played_songs_users_selected

def plot_kmean(sparse_ratings, k_min=2, k_max=50, k_step=4):
    sse = []
    for k in range(k_min, k_max, k_step ):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(sparse_ratings)
        sse.append(kmeans.inertia_)

    error_data = pd.DataFrame({'k':range(k_min, k_max, k_step), 'sse':sse})
    px.line(data_frame=error_data, x='k', y='sse')

def heatmap_cluster_groups(results, num_songs=40, num_users=30):
    for cluster in results['group'].unique():
        d = results[results['group'] == cluster].drop(['index', 'group'], axis=1)
        most_played_songs = find_most_played_songs(d, num_songs)
        most_interacted_users = find_most_interacted_users(d, num_users)
        d = d.reindex(d.mean().sort_values(ascending=False).index, axis=1)
        d = d.reindex(d.count(axis=1).sort_values(ascending=False).index)
        d = d.iloc[:, 1:]
        
        print('cluster {} with users {}'.format(cluster, d.shape[0]))
        
        figure = plt.figure(figsize=(20, 4), facecolor='gray')
        sns.heatmap(d)



def recommendations_cosine_model_1(user_id_index, num_of_songs, dense_matrix):
    most_similar_users = similar_users(user_id_index, dense_matrix)  #[0]
    # song ids(columns)  this user_id has reviewed
    song_ids = set(np.nonzero(dense_matrix[user_id_index])[1])
    
    recommendations = []
    
    observed_interactions = song_ids.copy()
    for similar_user in most_similar_users:
        if len(recommendations) < num_of_songs:
            # song ids(columns)  similar user has reviewed
            similar_user_song_ids = set(np.nonzero(dense_matrix[similar_user])[1])
            #add only song(columns) from similar user that have not already been reviewed by user_id
            recommendations.extend(list(similar_user_song_ids.difference(observed_interactions)))
            # fill user_ids blank reviews with rating from highest similar user you can find
            observed_interactions = observed_interactions.union(similar_user_song_ids)
        else:
            break            
        # song_recommendations = 
    return recommendations[:num_of_songs]

def perform_svd_test(X_train, X_val, X_test):
    #create interaction matrix(user_id, song_id, play_count) for each of train,validate and test sets
    global im_train, im_val, im_test
    im_train = create_interactions_matrix(X_train)
    im_val   = create_interactions_matrix(X_val)
    im_test  = create_interactions_matrix(X_test)


    #1. Find common (user_id & song_id) between train,validate and test sets to perform evaluation:
    #1a.find common indexes(user_ids -- rows) in [(train & validate), (train & test)] 
    train_idx = set(im_train.index)
    val_idx   = set(im_val.index)
    test_idx  = set(im_test.index)
    match_idx_train_val  = train_idx.intersection(val_idx)
    match_idx_train_test = train_idx.intersection(test_idx)

    #1b.find common song_ids(columns) in [(train & validate), (train & test)] 
    train_songs = set(im_train.columns)
    val_songs   = set(im_val.columns)
    test_songs  = set(im_test.columns)
    match_cols_train_val  = train_songs.intersection(val_songs)
    match_cols_train_test = train_songs.intersection(test_songs)

    #1c.list of common user_ids/song_ids cells in [train & validate], [train & test]
    im_common_train_val  = im_val.loc[match_idx_train_val,   match_cols_train_val]
    im_common_train_test = im_test.loc[match_idx_train_test, match_cols_train_test]
    im_common_train_val.shape, im_common_train_test.shape

    # SVD on im_train matrix to find s_train to use on validate and test sets.
    # s_train will be used in evaluating the model on validate & test sets
    u_train, s_train, vt_train = np.linalg.svd(im_train, full_matrices=False)

    #build validation set u, vt
    row_idxs = im_train.index.isin(val_idx)
    col_idxs = im_train.columns.isin(val_songs)
    u_val    = u_train[row_idxs, :]
    vt_val   = vt_train[:, col_idxs]

    #build test set u, vt
    row_idxs = im_train.index.isin(test_idx)
    col_idxs = im_train.columns.isin(test_songs)
    u_test   = u_train[row_idxs, :]
    vt_test  = vt_train[:, col_idxs]

    latent_features = np.arange(0, 710, 20)
    train_error = []
    val_error = []
    test_error = []

    for k in latent_features:
        u_train_lat, s_train_lat, vt_train_lat = u_train[:, :k], np.diag(s_train[:k]), vt_train[:k, :]
        u_val_lat,   vt_val_lat = u_val[:, :k],  vt_val[:k, :]
        u_test_lat, vt_test_lat = u_test[:, :k], vt_test[:k, :]
        
        im_train_preds = np.around(np.dot(np.dot(u_train_lat, s_train_lat), vt_train_lat))
        im_val_preds   = np.around(np.dot(np.dot(u_val_lat,   s_train_lat),   vt_val_lat))
        im_test_preds  = np.around(np.dot(np.dot(u_test_lat,  s_train_lat), vt_test_lat))
        
        train_difference = np.subtract(im_train, im_train_preds)
        val_difference   = np.subtract(im_val,   im_val_preds)
        test_difference  = np.subtract(im_test,  im_test_preds)
    
        error_train = np.sum(np.sum(np.abs(train_difference)))
        error_val   = np.sum(np.sum(np.abs(val_difference)))
        error_test  = np.sum(np.sum(np.abs(test_difference)))
    
        train_error.append(error_train)
        val_error.append(error_val)
        test_error.append(error_test)
        
    plt.figure(facecolor='gray')    
    plt.plot(latent_features, 1 - np.array(train_error)/(im_train.shape[0]*im_test.shape[1]), label='Train')
    plt.plot(latent_features, 1 - np.array(val_error) / (im_val.shape[0]*im_val.shape[1]),    label='Validate')
    plt.plot(latent_features, 1 - np.array(test_error)/ (im_test.shape[0]*im_test.shape[1]),  label='Test')
    plt.xlabel('Number of Latent Features')
    plt.ylabel('Accuracy')
    plt.legend()

    return u_train, s_train, vt_train, u_val, vt_val, u_test, vt_test

def recommendations_svd_model_2(user_index, interactions_matrix, preds_df, num_recommendations):
      
    # user_idx = user_index-1 # index starts at 0
    sorted_user_ratings = interactions_matrix[interactions_matrix.index== user_index] #sorted_user_ratings
    sorted_user_predictions = preds_df[preds_df.index ==user_index]

    x1= sorted_user_predictions.unstack().reset_index(name='user_predictions')
    x1.drop(['user_id'], axis=1,inplace=True)
    x2= sorted_user_ratings.unstack().reset_index(name='user_ratings')
    x2.drop(['user_id'], axis=1,inplace=True)
    temp = pd.concat( [x1,x2]  ,axis=1) 
    temp= temp[['song_id','user_ratings','user_predictions']]
    temp.sort_values(by='user_predictions', ascending=False)
    temp.reset_index(drop=True, inplace=True)
    temp=temp.iloc[:,1:]
    temp.rename(columns={'song_id':'Recommended Items'}, inplace=True)
    temp.set_index(['Recommended Items'],inplace=True)

    temp.index.name = 'Recommended Items'

    # temp.columns = ['user_ratings', 'user_predictions']

    temp = temp.loc[temp.user_ratings == 0]   
    temp = temp.sort_values('user_predictions', ascending=False)
    print('\nBelow are the recommended items for user(user_index = {}):\n'.format(user_index))
    print(temp.head(num_recommendations))
    return temp.head(num_recommendations)



# OTHER SOLUTION IN PROGRESS for k-mean clustering ???
# most_played_songs_users_im = most_played_songs_users_selected(dfu, dfs, num_users=1000, num_songs=1000)
# figure = plt.figure(figsize=(20, 40), facecolor='gray')
# sns.heatmap(most_played_songs_users_im)
# def get_song_titles_from_ids(recommended_song_ids, dfs):
#     recommended_song_titles = [] 
#     for song_id in recommended_song_ids:
#          recommended_song_titles.extend(
#                      dfs.query("song_id== @song_id")
#                         ['title'])

#     return recommended_song_titles
# def most_played_songs_users_selected(dfu, dfs, num_users=1000, num_songs=1000):
#     dfs_im = pd.DataFrame({'song_id':
#                                     dfu['song_id']
#                                     .value_counts()[:num_songs]
#                                     .index})

#     # Top <n>:100,000 Most Active Listeners: user_ids
#     dfu_im = (pd.DataFrame({'user_id':
#                                     dfu[dfu['song_id'].isin(dfs_im['song_id'])]
#                                     ['user_id']
#                                     .value_counts()[:num_users]
#                                     .index}))
#     # SMALLER INERACTION MATRIX- dfu_small
#     dfu_mini = dfs_im.merge(dfu, how='inner', on='song_id')
#     dfu_mini = dfu_im.merge(dfu_mini, how='inner', on='user_id')
#     dfu_mini = pd.merge(dfu_mini, dfs, how='left', on='song_id')
#     most_played_songs_users_im = create_interactions_matrix(dfu_mini)
#     return most_played_songs_users_im

##interactions_matrix = pd.pivot_table(dfu_small, index='user_id', columns= 'song_id', values='play_count')