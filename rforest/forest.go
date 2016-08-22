package rforest

import (
	"math"
	"math/rand"
	"time"
)

type Forest struct {
	trees []*Tree
}

func RandomForest(X [][]interface{}, y []interface{}, max_trees, max_samples, max_features int) *Forest {
	rand.Seed(time.Now().UnixNano())
	forest := &Forest{}
	forest.trees = make([]*Tree, max_trees)
	ch := make(chan bool)
	for i := 0; i < max_trees; i++ {
		go func(x int) {
			forest.trees[x] = BuildTree(X, y, max_samples, max_features)
			ch <- true
		}(i)
	}
	for i := 0; i < max_trees; i++ {
		<-ch
	}
	return forest
}

func DefaultRandomForest(X [][]interface{}, y []interface{}, max_trees int) *Forest {
	max_samples := int(math.Sqrt(float64(len(X))) + 0.5)
	max_features := int(math.Sqrt(float64(len(X[0]))) + 0.5)
	return RandomForest(X, y, max_trees, max_samples, max_features)
}

func (f Forest) Predict(X [][]interface{}) []interface{} {
	y := []interface{}{}
	for _, s := range X {
		mp := map[interface{}]float64{}
		for _, tree := range f.trees {
			classes := tree.root.Predict(s)
			for class, p := range classes {
				mp[class] += p
			}
		}
		ma := 0.0
		var class interface{}
		for k, v := range mp {
			if v > ma {
				ma = v
				class = k
			}
		}
		y = append(y, class)
	}
	return y
}
