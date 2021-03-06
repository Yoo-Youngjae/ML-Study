import pandas as pd
from sklearn.decomposition import TruncatedSVD
import numpy as np

rating = pd.read_csv('data/rating.csv')
movie = pd.read_csv('data/movie.csv')
print('rating.shape', rating.shape)
# 너무 커서 user 의 수를 좀 줄임..
rating = rating.loc[:500000]
rating.drop('timestamp', axis=1, inplace=True)
movie.drop('genres', axis=1, inplace=True)

#movieId 에 따른 user 의 평점들 모두 메겨짐
merge_data = pd.merge(rating, movie, on='movieId')
# pivot table 로 만듦. 한 user 가 내린 평점을 모두 1 row 안에서 나열. 평가 안한 영화는 0점
merge_data = merge_data.pivot_table('rating', index='userId', columns='title').fillna(0)
print('merge_data.shape', merge_data.shape)

# merge_data.shape = 702, 8225 == 사람수, 영화수
trans_merge_data = merge_data.T
# trans_merge_data.shape == 8225, 702
SVD = TruncatedSVD(n_components=12)
matrix = SVD.fit_transform(trans_merge_data)
# matrix.shape == 8225,12

corr = np.corrcoef(matrix)
# corr.shape == 9064, 9064
movie_title = merge_data.columns
movie_title_list = list(movie_title)
coffey_hands = movie_title_list.index("Avengers, The (2012)")

corr_coffey_hands = corr[coffey_hands]

score_list = list(corr_coffey_hands[(corr_coffey_hands >= 0.9)])
movie_final_list = list(movie_title[(corr_coffey_hands >= 0.9)])

final_df = pd.DataFrame({'title': movie_final_list, 'corr_score': score_list})
final_df = final_df.sort_values(by=['corr_score'], axis=0, ascending=False)
print(final_df.head(10))