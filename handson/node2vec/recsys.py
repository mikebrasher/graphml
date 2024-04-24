import pandas as pd
import networkx as nx
from collections import defaultdict
from node2vec import Node2Vec


def main():

    ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                          names=['user_id', 'movie_id', 'rating', 'unix_timestamp'])

    ratings = ratings[ratings.rating > 3]
    # ratings = ratings[ratings.movie_id < 50]

    movies = pd.read_csv('ml-100k/u.item', sep='|', usecols=[0, 1],
                         names=['movie_id',  'title'], encoding='latin-1')

    pairs = defaultdict(int)
    for group in ratings.groupby("user_id"):
        user_movies = list(group[1]["movie_id"])
        for i in range(len(user_movies)):
            for j in range(i+1, len(user_movies)):
                pairs[(user_movies[i], user_movies[j])] += 1

    graph = nx.Graph()
    connection_threshold = 20
    for p in pairs:
        movie1, movie2 = p
        score = pairs[p]
        if score >= connection_threshold:
            graph.add_edge(movie1, movie2, weight=score)

    walks = Node2Vec(graph, dimensions=64, walk_length=20, num_walks=200, p=2, q=1, workers=4)
    model = walks.fit(window=10, min_count=1, batch_words=4)

    def recommend(title):
        movie_str = str(movies[movies.title == title].movie_id.values[0])
        for similar_id, similarity in model.wv.most_similar(movie_str)[:5]:
            similar_title = movies[movies.movie_id == int(similar_id)].title.values[0]
            print('{}: {:.2f}'.format(similar_title, similarity))

    recommend('Star Wars (1977)')
    # recommend('Toy Story (1995)')


if __name__ == '__main__':
    main()
