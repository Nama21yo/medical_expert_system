;; ProbabiListic Logic Networks (PLN) Inference Engine - Clean Version
;; Medical Diagnosis System with Forward and Backward Chaining

! (register-module! ../symbolic)
! (import! &self symbolic:knowledge_base)

;; =============================================================================
;; UTILITY FUNCTIONS
;; =============================================================================

; (= (length ()) 0)
; (= (length ($head . $tail)) (+ 1 (length $tail)))

; (= (append () $List) $List)
; (= (append ($head . $tail) $List) (cons $head (append $tail $List)))

(= (max $a $b) (if (> $a $b) $a $b))
(= (min $a $b) (if (< $a $b) $a $b))

;; =============================================================================
;; TRUTH VALUE OPERATIONS
;; =============================================================================

(= (strength (TV $s $c)) $s)
(= (confidence (TV $s $c)) $c)
(= (make-tv $s $c) (TV $s $c))

;; Safe truth value extraction with defaults
(= (safe-tv $matches $default-tv)
   (if (> (size-atom $matches) 0) 
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
   (let* (
      ($all-symptoms (match &self (Evaluation (Predicate hasSymptom) (List $patient $symptom) $tv) $symptom))
          ($disease-inferences (forward-chain-symptom $patient $all-symptoms))
           (() (println! ( "DEBUG: given forward chain disease=" $disease-inferences )))
            ($d_tv (add-inferences-to-kb $patient $disease-inferences))
          )
          $d_tv
     )
     )


;; Forward chain from single symptom to diseases
(= (forward-chain-symptom $patient $symptom)
   (let* (($symptom-matches (match &self (Evaluation (Predicate hasSymptom) (List $patient $symptom) $tv) $tv))
          ($symptom-tv (safe-tv $symptom-matches (TV 0.0 0.0)))
          ($disease-causes (match &self (Evaluation (Predicate causes) (List $disease $symptom) $cause-tv) 
                                 (List $disease $cause-tv))))
     (infer-diseases-from-symptom $patient $symptom $symptom-tv $disease-causes)))

;; Infer diseases from symptom evidence
(= (infer-diseases-from-symptom $patient $symptom $symptom-tv ())
   ())
(= (infer-diseases-from-symptom $patient $symptom $symptom-tv (List $disease $cause-tv))
   (let* (($prior-matches (match &self (Evaluation (Predicate prevalence) (List $disease) $tv) $tv))
          ($disease-prior (safe-tv $prior-matches (TV 0.01 0.5)))
          ($inferred-tv (pln-abduction $symptom-tv $cause-tv $disease-prior))
          ($current-inference (List $disease $inferred-tv))
         ;  ($rest-inferences (infer-diseases-from-symptom $patient $symptom $symptom-tv $rest))
          )
     $current-inference ))

;; Add inferences to knowledge base
(= (add-inferences-to-kb $patient ())
   ())
(= (add-inferences-to-kb $patient (List $disease $tv) )
   (let* (
      (() (println! ( "DEBUG: given add inference=" $disease $tv)))
      ; ! (match &self (Evaluation (Predicate possibleDisease) (List TestPatient $disease) $tv) 
                                 ;  (List $disease $tv))
        (() (add-atom &self (Evaluation (Predicate possibleDisease) (List $patient $disease) $tv)))
      ; ($existing-matches (match &self (Evaluation (Predicate possibleDisease) (List $patient $disease) $existing-tv) $existing-tv))
      ; ($final-tv (if (> (size-atom $existing-matches) 0)
      ;                    (pln-revision $existing-tv $tv)
      ;                    $tv))
         )
   ;   (chain (() (remove-atom &self (Evaluation (Predicate possibleDisease) (List $patient $disease)) $existing-tv))
           
   ;          )
   (List $disease $tv)
      ))

;; =============================================================================
;; BACKWARD CHAINING IMPLEMENTATION
;; =============================================================================

;; Backward chain to find evidence for a specific disease
(= (backward-chain $patient $target-disease)
   (let* (($prior-matches (match &self (Evaluation (Predicate prevalence) (List $target-disease) $tv) $tv))
          ($disease-prior (safe-tv $prior-matches (TV 0.01 0.5)))
          ($supporting-symptoms (match &self (Evaluation (Predicate causes) (List $target-disease $symptom) $cause-tv) 
                                      (List $symptom $cause-tv)))
          ($combined-evidence (combine-symptom-evidence $patient $target-disease $disease-prior $supporting-symptoms)))
     $combined-evidence))

;; Combine evidence from multiple supporting symptoms
(= (combine-symptom-evidence $patient $disease $base-tv ())
   $base-tv)
(= (combine-symptom-evidence $patient $disease $base-tv (List $symptom $cause-tv))
   (let* (($symptom-matches (match &self (Evaluation (Predicate hasSymptom) (List $patient $symptom) $tv) $tv))
          ($symptom-tv (safe-tv $symptom-matches (TV 0.0 0.0)))
          ($evidence-strength (if (> (strength $symptom-tv) 0)
                                 (pln-conjunction $symptom-tv $cause-tv)
                                 (TV 0.0 0.0)))
          ($revised-tv (if (> (strength $evidence-strength) 0)
                          (pln-revision $base-tv $evidence-strength)
                          $base-tv))
         ;  ($final-tv (combine-symptom-evidence $patient $disease $revised-tv $rest))
          )
     $revised-tv))

;; =============================================================================
;; MAIN DIAGNOSIS FUNCTIONS
;; =============================================================================

;; Complete diagnosis using forward chaining
(= (diagnose-forward $patient)
   (let* (($forward-results (forward-chain-all $patient))
             (() (println! ( "DEBUG: given risk value=" $forward-results  )))
          ($final-diagnoses (match &self (Evaluation (Predicate possibleDisease) (List $patient $disease) $tv) 
                                  (List $disease $tv)))
             (() (println! ( "DEBUG: given final diagnoses=" $final-diagnoses  )))
                  
                  )
     $final-diagnoses))

;; Targeted diagnosis using backward chaining
(= (diagnose-backward $patient $target-disease)
   (let* (
      ($backward-result (backward-chain $patient $target-disease))
      (() (add-atom &self (Evaluation (Predicate possibleDisease) (List $patient $target-disease) $backward-result)))
   )
   ;   (chain !(add-atom &self (Evaluation (Predicate possibleDisease) (List $patient $target-disease) $backward-result))
   ;          $backward-result)
            $backward-result
            )
      )

;; Get diagnostic explanation
(= (explain-diagnosis $patient $disease)
   (let* (($diag-matches (match &self (Evaluation (Predicate possibleDisease) (List $patient $disease) $tv) $tv))
          ($diagnosis-tv (safe-tv $diag-matches (TV 0.0 0.0)))
          ($supporting-symptoms (match &self (Evaluation (Predicate hasSymptom) (List $patient $symptom) $s-tv)
                                      (match &self (Evaluation (Predicate causes) (List $disease $symptom) $c-tv)
                                             (List $symptom $s-tv $c-tv))))
          ($prevalence-matches (match &self (Evaluation (Predicate prevalence) (List $disease) $tv) $tv))
          ($prevalence-tv (safe-tv $prevalence-matches (TV 0.01 0.5))))
     (List (cons diagnosis $diagnosis-tv) 
           (cons symptoms $supporting-symptoms) 
           (cons prevalence $prevalence-tv))))

;; =============================================================================
;; TEST SETUP
;; =============================================================================

;; Setup test patient
!(add-atom &self (Evaluation (Predicate hasSymptom) (List TestPatient ChestPain) (TV 0.9 0.95)))
!(add-atom &self (Evaluation (Predicate hasSymptom) (List TestPatient ShortnessOfBreath) (TV 0.8 0.90)))
!(add-atom &self (Evaluation (Predicate hasSymptom) (List TestPatient Sweating) (TV 0.7 0.85)))

; ;; Test forward chaining
; !(forward-chain-all TestPatient)

; ; Test backward chaining
; !(backward-chain TestPatient MyocardialInfarction)
; !(backward-chain Patient_da3c7b0c-a469-42e0-ba0b-fad094e72fb6 MyocardialInfarction)
; !(backward-chain TestPatient pneumonia)

; ; Test diagnosis functions

; !(diagnose-forward TestPatient)
; !(diagnose-backward TestPatient MyocardialInfarction)

