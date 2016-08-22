package rforest

import (
	//"fmt"
	"math/rand"
	"reflect"
	"sort"
)

const (
	eps = 1e-6
)

type TreeNode struct {
	value    interface{}             //节点划分的阈值
	col      int                     //节点按col列划分
	col_Type string                  //col列的类型(连续或离散)
	lchild   *TreeNode               //左孩子节点
	rchild   *TreeNode               //右孩子节点
	classes  map[interface{}]float64 //节点包含的样本类别信息
}

type Tree struct {
	root *TreeNode
}

type sample struct {
	features []interface{} //属性值
	class    interface{}   //类别
}

type samples []sample

type samplesWrapper struct {
	s     samples
	cmpby int
}

func (sw samplesWrapper) Len() int {
	return len(sw.s)
}

func (sw samplesWrapper) Less(i, j int) bool {
	x, y := sw.s[i].features[sw.cmpby], sw.s[j].features[sw.cmpby]
	switch x.(type) {
	case string:
		return x.(string) < y.(string)
	case float64:
		return x.(float64) < y.(float64)
	default:
		return false
	}
}

func (sw samplesWrapper) Swap(i, j int) {
	sw.s[i], sw.s[j] = sw.s[j], sw.s[i]
}

func getRandom(n int, m int, bootstrap bool) []int {
	rg := []int{}
	if bootstrap {
		for i := 0; i < m; i++ {
			rg = append(rg, rand.Intn(n))
		}
		return rg
	} else {
		for i := 0; i < n; i++ {
			rg = append(rg, i)
		}
		for i := 0; i < m; i++ {
			j := rand.Intn(n - i)
			rg[i], rg[j] = rg[j], rg[i]
		}
		return rg[:m]
	}
}

func gini(data samples) float64 {
	l := float64(len(data))
	mp := make(map[interface{}]float64)
	for _, sample := range data {
		mp[sample.class]++
	}
	ret := 1.0
	for _, value := range mp {
		ret -= (value / l) * (value / l)
	}
	return ret
}

func buildTreeNode(data samples, max_features int) *TreeNode {
	features := getRandom(len(data[0].features), max_features, false)
	best_gain := 1.0
	best_l := samples{}
	best_r := samples{}
	var best_value interface{}
	best_type := "consecutive"
	best_col := 0
	for _, col := range features {
		mp := map[interface{}]int{}
		for _, s := range data {
			mp[s.features[col]]++
		}
		//连续属性
		if len(mp) > 10 && reflect.TypeOf(data[0].features[col]).String() != "string" {
			//按该属性排序
			sort.Sort(samplesWrapper{data, col})
			for i := 0; i < len(data)-1; i++ {
				if data[i].class != data[i+1].class {
					l, r := data[:i+1], data[i+1:]
					lgini, rgini := gini(l), gini(r)
					gini_gain := lgini*float64(len(l))/float64(len(data)) + rgini*float64(len(r))/float64(len(data))
					if gini_gain < best_gain {
						best_gain = gini_gain
						best_l, best_r = l, r
						best_value = data[i].features[col]
						best_col = col
						best_type = "consecutive"
					}
				}
			}
		} else {
			for k, _ := range mp {
				l, r := samples{}, samples{}
				for _, s := range data {
					if s.features[col] == k {
						l = append(l, s)
					} else {
						r = append(r, s)
					}
				}
				lgini, rgini := gini(l), gini(r)
				gini_gain := lgini*float64(len(l))/float64(len(data)) + rgini*float64(len(r))/float64(len(data))
				if gini_gain < best_gain {
					best_gain = gini_gain
					best_l, best_r = l, r
					best_value = k
					best_col = col
					best_type = "discrete"
				}
			}
		}
	}

	node := &TreeNode{}
	if gini(data)-best_gain <= eps {
		node.classes = map[interface{}]float64{}
		for _, s := range data {
			node.classes[s.class]++
		}
		for k, _ := range node.classes {
			node.classes[k] /= float64(len(data))
		}
		return node
	}
	node.col = best_col
	node.col_Type = best_type
	node.value = best_value
	node.lchild = buildTreeNode(best_l, max_features)
	node.rchild = buildTreeNode(best_r, max_features)

	return node
}

func BuildTree(X [][]interface{}, y []interface{}, max_samples, max_features int) *Tree {
	data := samples{}
	for i := 0; i < len(y); i++ {
		data = append(data, sample{X[i], y[i]})
	}
	//随机选择样本
	tmp := getRandom(len(data), max_samples, true)
	select_datas := samples{}
	for i := 0; i < max_samples; i++ {
		select_datas = append(select_datas, data[tmp[i]])
	}

	tree := &Tree{}
	tree.root = buildTreeNode(select_datas, max_features)
	return tree
}

func (t *TreeNode) Predict(sample []interface{}) map[interface{}]float64 {
	if t.classes != nil {
		return t.classes
	}
	if t.col_Type == "consecutive" {
		flag := true
		switch t.value.(type) {
		case float64:
			flag = sample[t.col].(float64) <= t.value.(float64)
		}
		if flag {
			return t.lchild.Predict(sample)
		} else {
			return t.rchild.Predict(sample)
		}
	} else {
		if sample[t.col] == t.value {
			return t.lchild.Predict(sample)
		} else {
			return t.rchild.Predict(sample)
		}
	}
}
