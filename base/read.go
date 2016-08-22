package base

import (
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

func Read_txt(filepath string, split string) [][]interface{} {
	f, _ := os.Open(filepath)
	defer f.Close()
	stream, _ := ioutil.ReadAll(f)
	content := string(stream)
	lines := strings.Split(content, "\n")
	data := [][]interface{}{}
	for _, line := range lines {
		line = strings.TrimRight(line, "\r\n")
		if len(line) == 0 {
			continue
		}
		tmp := []interface{}{}
		cols := strings.Split(line, split)
		for _, col := range cols {
			f, err := strconv.ParseFloat(col, 64)
			if err == nil {
				tmp = append(tmp, f)
			} else {
				tmp = append(tmp, col)
			}

		}
		data = append(data, tmp)
	}
	return data
}
