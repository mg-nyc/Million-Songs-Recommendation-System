# SongsRecSys
# Million Songs Problem Statement
#
## Context
With the advent of technology, societies have become more efficient with their lives. But at the same time, individual human lives have become much more fast paced and distracted by leaving little time to explore artistic pursuits. Also, the technology has made significant advancements in the ability to coexist with art and general entertainment. In fact, it has made it easier for humans with shortage of time to find and consume good content. Therefore, one of the key challenges for the companies is to be able to figure out what kind of content their customers are most likely to consume. Almost every internet based company's revenue relies on the time consumers spend on their platforms. These companies need to be able to figure out what kind of content is needed in order to increase the time spent by customers on their platform and make their experience better.

Spotify is one such audio content provider who has got a huge market base across the world. It has grown significantly because of its ability to recommend the ‘best’ next song to each and every customer based on the huge preference database they have gathered over time like millions of customers and billions of songs. This is done by using smart recommendation systems that can recommend songs based on the users’ likes/dislikes

## Problem Statement

Build a recommendation system to propose the top 10 songs for a user based on the likelihood of listening to those songs.

## Data Dictionary
The core data is the Taste Profile Subset released by The Echo Nest as part of the Million Song Dataset. There are two files in this dataset. One contains the details about the song id, titles, release, artist name and the year of release. Second file contains the user id, song id and the play count of users.

## song_data
 1. song_id - A unique id given to every song
 2. title - Title of the song
 3. Release - Name of the released album
 4. Artist_name - Name of the artist
 5. year - Year of release

## count_data
 1. user _id - A unique id given to the user
 2. song_id - A unique id given to the song
 3. play_count - Number of times the song was played

## Data Source:  http://millionsongdataset.com/
