package benchmark

import "testing"

func TestFailureKind(t *testing.T) {
	for msg, want := range map[string]string{
		"TimeoutError: the model did not finish within the run timeout of 300 s":                    "timeout",
		"AnalysisTimeoutError: The LLM request timed out. Please try again.":                        "timeout",
		"UnreadableReceiptError: not a receipt":                                                     "unreadable",
		"ValueError: Failed to get a valid response from model 'm' after 3 attempts: total missing": "invalid_response",
		"ValueError: Model 'm' returned a message indicating it cannot process images.":             "invalid_response",
		"Receipt image changed since run creation":                                                  "inputs_changed",
		"Extraction prompt changed since run creation; create a new run":                            "inputs_changed",
		"ValueError: 500 internal error":                                                            "error",
		"":                                                                                          "error",
	} {
		if got := FailureKind(msg); got != want {
			t.Errorf("%q: %s, want %s", msg, got, want)
		}
	}
}
