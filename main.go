package main

import (
	"./base"
	"./rforest"
	"fmt"
)

func main() {
	train := base.Read_txt("iris2.data.txt", ",")
	X := [][]interface{}{}
	y := []interface{}{}
	for _, line := range train {
		X = append(X, line[:len(line)-1])
		y = append(y, line[len(line)-1])
	}
	rf := rforest.DefaultRandomForest(X, y, 1000)
	predict_y := rf.Predict(X)
	right := 0
	for i := 0; i < len(y); i++ {
		if predict_y[i] == y[i] {
			right++
		}
	}
	fmt.Println(float64(right) / float64(len(y)))
}
