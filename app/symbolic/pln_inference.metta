;; ProbabiListic Logic Networks (PLN) Inference Engine - Corrected MeTTa Version
;; Medical Diagnosis System with Proper Forward and Backward Chaining

! (register-module! ../symbolic)
! (import! &self symbolic:knowledge_base)

;; =============================================================================
;; UTILITY FUNCTIONS
;; =============================================================================

;; List operations
(= (length ()) 0)
(= (length ($head . $tail)) (+ 1 (length $tail)))

(= (car ($head . $tail)) $head)
(= (cdr ($head . $tail)) $tail)

(= (cons $head $tail) ($head . $tail))

(= (append () $List) $List)
(= (append ($head . $tail) $List) (cons $head (append $tail $List)))

;; Mathematical operations
(= (max $a $b) (if (> $a $b) $a $b))
(= (min $a $b) (if (< $a $b) $a $b))

;; =============================================================================
;; TRUTH VALUE OPERATIONS
;; =============================================================================

;; Extract strength from Truth Value
(= (strength (TV $s $c)) $s)

;; Extract confidence from Truth Value  
(= (confidence (TV $s $c)) $c)

;; Create Truth Value
(= (make-tv $s $c) (TV $s $c))

;; Safe truth value extraction with defaults
(= (safe-tv $matches $default-tv)
   (if (> (size-atom $matches) 0) 
    ;    (car-atom $matches) 
        $matches
        $default-tv))

;; =============================================================================
;; PLN INFERENCE RULES
;; =============================================================================

;; PLN Abduction Rule: P(Disease|Symptom) from P(Symptom|Disease), P(Symptom), P(Disease)
(= (pln-abduction $symptom-tv $cause-tv $prior-tv)
   (let* (($s-symptom (strength $symptom-tv))
          ($c-symptom (confidence $symptom-tv))
          ($s-cause (strength $cause-tv))
          ($c-cause (confidence $cause-tv))
          ($s-prior (strength $prior-tv))
          ($c-prior (confidence $prior-tv))
          ($numerator (* $s-prior (* $s-cause $s-symptom)))
          ($posterior-strength (min 1.0 (if (> $s-symptom 0) 
                                            (/ $numerator $s-symptom) 
                                            $numerator)))
          ($posterior-confidence (* $c-symptom (* $c-cause $c-prior))))
     (TV $posterior-strength $posterior-confidence)))

;; PLN Revision Rule: Combine multiple evidence sources
(= (pln-revision $tv1 $tv2)
   (let* (($s1 (strength $tv1))
          ($c1 (confidence $tv1))
          ($s2 (strength $tv2))
          ($c2 (confidence $tv2))
          ($total-weight (+ $c1 $c2))
          ($new-strength (if (> $total-weight 0) 
                            (/ (+ (* $s1 $c1) (* $s2 $c2)) $total-weight) 
                            0))
          ($new-confidence (min 1.0 (- 1.0 (* (- 1.0 $c1) (- 1.0 $c2))))))
     (TV $new-strength $new-confidence)))

;; PLN Conjunction: Combine multiple symptoms for same disease
(= (pln-conjunction $tv1 $tv2)
   (let* (($s1 (strength $tv1))
          ($c1 (confidence $tv1))
          ($s2 (strength $tv2))
          ($c2 (confidence $tv2))
          ($conj-strength (* $s1 $s2))
          ($conj-confidence (* $c1 $c2)))
     (TV $conj-strength $conj-confidence)))

;; =============================================================================
;; FORWARD CHAINING IMPLEMENTATION
;; =============================================================================

;; Forward chain from all symptoms to possible diseases
(= (forward-chain-all $patient)
    (let* (($all-symptoms (match &self (Evaluation (Predicate hasSymptom) (List $patient $symptom) $tv) $symptom))
           ($symptom-results (flat-forward-chain-symptom $patient $all-symptoms))
           ($disease_and_tv (cdr-atom $symptom-results))
           ($disease (car-atom  $disease_and_tv ))
        ;    (() (println! (string-append "DEBUG: added disease=" $disease)))
           (!(add-atom &self (Evaluation (Predicate possibleDisease) (List $patient $disease) $tv)))
           )
      $symptom-results))
; (= (return-disease (List $disease $tv) )
;     (let $disease (List ))
; )
;; Flattened map-symptoms: returns a flat List of (disease, TV) pairs
(= (flat-map-symptoms $patient ()) ())
(= (flat-map-symptoms $patient ($symptom . $rest))
    (let* (($current-results (flat-forward-chain-symptom $patient $symptom))
           ($rest-results (flat-map-symptoms $patient $rest)))
      (append $current-results $rest-results)))

;; Flat version of forward-chain-symptom: returns a flat List of (disease, TV) pairs
(= (flat-forward-chain-symptom $patient $symptom)
    (let* (($symptom-matches (match &self (Evaluation (Predicate hasSymptom) (List $patient $symptom) $tv) $tv))
            (() (println! (string-append "DEBUG: given syptom tv=" $symptom-matches)))
           ($symptom-tv (safe-tv $symptom-matches (TV 0.0 0.0)))
           (() (println! (string-append "DEBUG: flat-forward-chain-symptom: symptom=" $symptom " tv=" $symptom-tv)))
           ($disease-causes (match &self (Evaluation (Predicate causes) (List $disease $symptom) $cause-tv) 
                                  (List $disease $cause-tv)))
           (() (println! (string-append "DEBUG: flat-forward-chain-symptom: disease-causes=" $disease-causes)))
           ($disease-inferences (flat-infer-disease-from-symptom $patient $symptom $symptom-tv $disease-causes)))
      $disease-inferences))

;; Flat version of map-disease-causes: returns a flat List of (disease, TV) pairs
(= (flat-map-disease-causes $patient $symptom $symptom-tv ()) ())
                                                            ; (List MyocardialInfarction (TV 0.95 0.98))
(= (flat-map-disease-causes $patient $symptom $symptom-tv (List $disease $cause-tv))
    (let* (
         (() (println! (string-append "DEBUG: flat-map-disease-causes: disease=" $disease " cause-tv=" $cause-tv)))
        ($current-inference (flat-infer-disease-from-symptom $patient $disease $symptom $symptom-tv $cause-tv))
        (() (println! (string-append "DEBUG: flat-map-disease-causes: current-inference=" $current-inference)))
           ($rest-inferences (flat-map-disease-causes $patient $symptom $symptom-tv $rest))
           )
      (cons $current-inference $rest-inferences)))

;; Flat version of infer-disease-from-symptom: returns (disease, TV) pair
(= (flat-infer-disease-from-symptom $patient $symptom $symptom-tv (List $disease $cause-tv))
    (let* (($prior-matches (match &self (Evaluation (Predicate prevalence) (List $disease) $tv) $tv))
           ($disease-prior (safe-tv $prior-matches (TV 0.01 0.5)))
           ($inferred-tv (pln-abduction $symptom-tv $cause-tv $disease-prior)))
      (List $disease $inferred-tv)))

;; =============================================================================
;; BACKWARD CHAINING IMPLEMENTATION
;; =============================================================================

;; Backward chain to find evidence for a specific disease
(= (backward-chain $patient $target-disease)
   (let* (
          ($prior-matches (match &self (Evaluation (Predicate prevalence) (List $target-disease) $tv) $tv))
           (() (println! (string-append "DEBUG: prior matches=" $prior-matches)))
          ($disease-prior (safe-tv $prior-matches (TV 0.01 0.5)))
           (() (println! (string-append "DEBUG: given disease prior=" $disease-prior)))
          ($supporting-symptoms (match &self (Evaluation (Predicate causes) (List $target-disease $symptom) $cause-tv) 
                                      (List $symptom $cause-tv)))
          (() (println! (string-append "DEBUG: given supporting=" $supporting-symptoms)))
          ($symptom-evidence (evaluate-symptom-evidence $patient $target-disease $supporting-symptoms))
           (() (println! (string-append "DEBUG: given syptom evidence=" $symptom-evidence)))
          ($combined-evidence (combine-evidence-List $disease-prior $symptom-evidence))
           (() (println! (string-append "DEBUG: given combined =" $combined-evidence)))
          )
     $combined-evidence))

;; Evaluate evidence from symptoms
(= (evaluate-symptom-evidence $patient $disease ())
   ())
(= (evaluate-symptom-evidence $patient $disease (List $symptom $cause-tv))
   (let* (
            ($symptom-matches (match &self (Evaluation (Predicate hasSymptom) (List $patient $symptom) $tv) $tv))
             (() (println! (string-append "DEBUG: given syptom matches=" $symptom-matches)))
          ($symptom-tv (safe-tv $symptom-matches (TV 0.0 0.0)))
           (() (println! (string-append "DEBUG: given syptom tv=" $symptom-tv)))
          ($evidence-strength (if (> (strength $symptom-tv) 0)
                                 (pln-conjunction $symptom-tv $cause-tv)
                                 (TV 0.0 0.0)))
             (() (println! (string-append "DEBUG: given evidence strength=" $evidence-strength)))
        ;   ($rest-evidence (evaluate-symptom-evidence $patient $disease $rest))
          )
     (List $symptom $evidence-strength)))

;; Combine evidence from multiple sources
(= (combine-evidence-List $base-tv ())
   $base-tv)
(= (combine-evidence-List $base-tv (List $symptom $evidence-tv))
   (let* (
            ($combined-tv (if (> (strength $evidence-tv) 0)
                                (pln-revision $base-tv $evidence-tv)
                                $base-tv)
                           )
           (() (println! (string-append "DEBUG: given combined tv=" $combined-tv))) 
        ;   ($final-tv (combine-evidence-List $combined-tv $rest))
          
          )
     $combined-tv))

;; =============================================================================
;; RISK FACTOR INTEGRATION
;; =============================================================================

;; Apply risk factor adjustments
(= (apply-risk-factors $patient)
   (let* (
        (() (println! (string-append "DEBUG: given current patient=" $patient))) 
    
        ($current-diagnoses (match &self (Evaluation (Predicate possibleDisease) (List $patient $disease) $tv) 
                                    (List $disease $tv)))
        (() (println! (string-append "DEBUG: given apply risks=" $current-diagnoses))) 
          ($adjusted-diagnoses (adjust-for-risk-factors $patient $current-diagnoses))
          )
     $adjusted-diagnoses)
     
     )

;; Apply risk adjustments to each diagnosis
(= (map-risk-adjustments $patient ())
   ())
(= (map-risk-adjustments $patient (($disease $base-tv) . $rest))
   (let* (($adjusted-tv (adjust-for-risk-factors $patient $disease $base-tv))
          ($rest-adjusted (map-risk-adjustments $patient $rest)))
     (cons (List $disease $adjusted-tv) $rest-adjusted)))

;; Adjust single disease for risk factors
(= (adjust-for-risk-factors $patient (List $disease $base-tv))
   (let* (($risk-factors (match &self (Evaluation (Predicate hasRiskFactor) (List $patient $rf) $rf-tv) $rf))
          ($risk-effects (map-risk-effects $patient $disease $risk-factors))
          ($adjusted-tv (apply-risk-effects $base-tv $risk-effects)))
     (chain !(remove-atom &self (Evaluation (Predicate possibleDisease) (List $patient $disease) $base-tv))
            $removed
            (chain !(add-atom &self (Evaluation (Predicate possibleDisease) (List $patient $disease) $adjusted-tv))
                   $added
                   $adjusted-tv))))

;; Map risk effects for disease
(= (map-risk-effects $patient $disease ())
   ())
(= (map-risk-effects $patient $disease ($rf . $rest))
   (let* (($rf-matches (match &self (Evaluation (Predicate hasRiskFactor) (List $patient $rf) $rf-tv) $rf-tv))
          ($rf-tv (safe-tv $rf-matches (TV 0.0 0.0)))
          ($increase-matches (match &self (Evaluation (Predicate increasesRisk) (List $rf $disease) $increase-tv) $increase-tv))
          ($increase-tv (safe-tv $increase-matches (TV 0.0 0.0)))
          ($risk-effect (calculate-risk-effect $rf-tv $increase-tv))
          ($rest-effects (map-risk-effects $patient $disease $rest)))
     (cons $risk-effect $rest-effects)))

;; Calculate risk effect
(= (calculate-risk-effect $rf-tv $increase-tv)
   (let* (($rf-strength (strength $rf-tv))
          ($increase-strength (strength $increase-tv))
          ($risk-multiplier (+ 1.0 (* $rf-strength (* $increase-strength 0.5)))))
     $risk-multiplier))

;; Apply risk effects to truth value
(= (apply-risk-effects $base-tv ())
   $base-tv)
(= (apply-risk-effects $base-tv ($multiplier . $rest))
   (let* (($base-strength (strength $base-tv))
          ($base-confidence (confidence $base-tv))
          ($adjusted-strength (min 1.0 (* $base-strength $multiplier)))
          ($intermediate-tv (TV $adjusted-strength $base-confidence))
          ($final-tv (apply-risk-effects $intermediate-tv $rest)))
     $final-tv))


;; =============================================================================
;; MAIN DIAGNOSIS FUNCTIONS
;; =============================================================================

;; Complete diagnosis using forward chaining
(= (diagnose-forward $patient)
   (let* (($forward-results (forward-chain-all $patient))
        (() (println! (string-append "DEBUG: given diag forward=" $forward-results)))
          ($risk-adjusted (apply-risk-factors $patient))
          (() (println! (string-append "DEBUG: risk adjusted=" $risk-adjusted)))
          ($final-diagnoses (match &self (Evaluation (Predicate possibleDisease) (List $patient $disease) $tv) 
                                  (List $disease $tv)))
        (() (println! (string-append "DEBUG: given final diag=" $final-diagnoses)))
                                  )
     $final-diagnoses))

;; Targeted diagnosis using backward chaining
(= (diagnose-backward $patient $target-disease)
   (let* (
            ($backward-result (backward-chain $patient $target-disease))
             (() (println! (string-append "DEBUG: given final diag=" $backward-result)))

            ; (!(add-atom &self (Evaluation (Predicate possibleDisease) (List $patient $target-disease) $backward-result)))
     (chain !(add-atom &self (Evaluation (Predicate possibleDisease) (List $patient $target-disease) $backward-result))
            $added
            $backward-result
            )
            )
             $backward-result
            )
            
        )

;; Combined diagnosis approach
(= (diagnose-combined $patient)
   (let* (($forward-results (diagnose-forward $patient))
          ($top-candidates (get-top-candidates $forward-results 3))
          ($backward-refined (map-backward-refinement $patient $top-candidates))
          ($final-results (match &self (Evaluation (Predicate possibleDisease) (List $patient $disease) $tv) 
                                (List $disease $tv))))
     $final-results))

;; Get top N candidates by strength
(= (get-top-candidates $diagnoses $n)
   (take $n $diagnoses))

(= (take 0 $List) ())
(= (take $n ()) ())
(= (take $n ($head . $tail))
   (if (> $n 0)
       (cons $head (take (- $n 1) $tail))
       ()))

;; Apply backward chaining refinement
(= (map-backward-refinement $patient ())
   ())
(= (map-backward-refinement $patient (($disease $tv) . $rest))
   (let* (($refined-tv (backward-chain $patient $disease))
          ($rest-refined (map-backward-refinement $patient $rest)))
     (cons (List $disease $refined-tv) $rest-refined)))

;; =============================================================================
;; DIAGNOSTIC EXPLANATION
;; =============================================================================

;; Get comprehensive explanation for diagnosis
(= (explain-diagnosis $patient $disease)
   (let* (($diag-matches (match &self (Evaluation (Predicate possibleDisease) (List $patient $disease) $tv) $tv))
          ($diagnosis-tv (safe-tv $diag-matches (TV 0.0 0.0)))
          ($supporting-symptoms (match &self (Evaluation (Predicate hasSymptom) (List $patient $symptom) $s-tv)
                                      (match &self (Evaluation (Predicate causes) (List $disease $symptom) $c-tv)
                                             (List $symptom $s-tv $c-tv))))
          ($risk-factors (match &self (Evaluation (Predicate hasRiskFactor) (List $patient $rf) $rf-tv)
                               (match &self (Evaluation (Predicate increasesRisk) (List $rf $disease) $inc-tv)
                                      (List $rf $rf-tv $inc-tv))))
          ($prevalence-matches (match &self (Evaluation (Predicate prevalence) (List $disease) $tv) $tv))
          ($prevalence-tv (safe-tv $prevalence-matches (TV 0.01 0.5))))
     (List (cons diagnosis $diagnosis-tv) 
           (cons symptoms $supporting-symptoms) 
           (cons risk-factors $risk-factors)
           (cons prevalence $prevalence-tv))))

;; =============================================================================
;; TEST SETUP AND EXECUTION
;; =============================================================================

;; Setup test patient with comprehensive data
!(add-atom &self (Evaluation (Predicate hasSymptom) (List TestPatient ChestPain) (TV 0.9 0.95)))
!(add-atom &self (Evaluation (Predicate hasSymptom) (List TestPatient ShortnessOfBreath) (TV 0.8 0.90)))
!(add-atom &self (Evaluation (Predicate hasSymptom) (List TestPatient Sweating) (TV 0.7 0.85)))
!(add-atom &self (Evaluation (Predicate hasRiskFactor) (List TestPatient AgeOver50) (TV 1.0 1.0)))
!(add-atom &self (Evaluation (Predicate hasRiskFactor) (List TestPatient Smoker) (TV 0.9 0.95)))

;; Test truth value operations
!(strength (TV 0.9 0.95))
!(confidence (TV 0.9 0.95))
!(make-tv 0.8 0.9)

;; Test PLN inference rules
!(pln-abduction (TV 0.9 0.95) (TV 0.95 0.98) (TV 0.02 0.95))
!(pln-revision (TV 0.5 0.8) (TV 0.6 0.85))
!(pln-conjunction (TV 0.8 0.9) (TV 0.7 0.85))

; ;; Test forward chaining
!(forward-chain-all TestPatient)

; ;; Add possible diseases as atoms for TestPatient

; (= (add-possible-diseases $patient ()) ())
; (= (add-possible-diseases $patient (List $disease $tv))
;    (chain !(add-atom &self (Evaluation (Predicate possibleDisease) (List $patient $disease) $tv))
;           $added
;           (add-possible-diseases $patient $rest)))
; !(add-possible-diseases TestPatient (forward-chain-all TestPatient))

; ;; Test backward chaining
; !(backward-chain TestPatient MyocardialInfarction)

; ; ;; Test risk factor application
!(apply-risk-factors TestPatient)

; ; ;; Complete diagnosis tests
; !(diagnose-forward TestPatient)
; !(diagnose-backward TestPatient MyocardialInfarction)
; !(diagnose-combined TestPatient)

; ;; Test explanation
; !(explain-diagnosis TestPatient MyocardialInfarction)

; ;; Show final knowledge state
!(match &self (Evaluation (Predicate possibleDisease) (List TestPatient $disease) $tv) (List $disease $tv))
