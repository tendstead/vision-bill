// Command vbctl bundles Vision Bill's operational tools:
//
//	vbctl benchmark       queue a benchmark run through the API
//	vbctl thumbnails      create thumbnails for images that have none
//	vbctl context-budget  measure Ollama prompt token usage
package main

import (
	"fmt"
	"os"
)

const usage = `usage: vbctl <command> [flags]

commands:
  benchmark       queue a benchmark run through the API (admin account)
  thumbnails      create thumbnails for images that do not have one yet
  context-budget  measure the Ollama token budget of the extraction prompt

Run "vbctl <command> -h" for the flags of a command.`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, usage)
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "benchmark":
		err = runBenchmark(os.Args[2:])
	case "thumbnails":
		err = runThumbnails(os.Args[2:])
	case "context-budget":
		err = runContextBudget(os.Args[2:])
	case "-h", "--help", "help":
		fmt.Println(usage)
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s\n", os.Args[1], usage)
		os.Exit(2)
	}
	if err != nil {
		if e, ok := err.(usageError); ok {
			fmt.Fprintln(os.Stderr, string(e))
			os.Exit(2)
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// usageError is a flag/argument problem (exit status 2).
type usageError string

func (e usageError) Error() string { return string(e) }

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
