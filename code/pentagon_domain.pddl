(define (domain pentagon)
  (:requirements :strips)
  
  (:predicates
    (on ?x ?y)
    (ontable ?x)
    (clear ?x)
    (holding ?x)
    (handempty)
    
    (pentagon-edge ?e)
    (layer ?l)
    (at-edge ?b ?e ?l)
    (edge-free ?e ?l)
    (edge-occupied ?e ?l)
    (in-layer ?b ?l)
  )
  
  (:action pick-up
    :parameters (?x)
    :precondition (and 
      (clear ?x)
      (ontable ?x)
      (handempty)
    )
    :effect (and
      (not (ontable ?x))
      (not (clear ?x))
      (not (handempty))
      (holding ?x)
    )
  )
  
  (:action put-down
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and
      (not (holding ?x))
      (clear ?x)
      (handempty)
      (ontable ?x)
    )
  )
  
  (:action place-at-edge
    :parameters (?b ?e ?l)
    :precondition (and 
      (holding ?b)
      (pentagon-edge ?e)
      (edge-free ?e ?l)
      (layer ?l)
    )
    :effect (and
      (not (holding ?b))
      (at-edge ?b ?e ?l)
      (edge-occupied ?e ?l)
      (not (edge-free ?e ?l))
      (in-layer ?b ?l)
      (clear ?b)
      (ontable ?b)
      (handempty)
    )
  )
  
  (:action pickup-from-edge
    :parameters (?b ?e ?l)
    :precondition (and
      (at-edge ?b ?e ?l)
      (clear ?b)
      (handempty)
      (pentagon-edge ?e)
      (layer ?l)
    )
    :effect (and
      (holding ?b)
      (not (at-edge ?b ?e ?l))
      (not (clear ?b))
      (not (handempty))
      (edge-free ?e ?l)
      (not (edge-occupied ?e ?l))
      (not (in-layer ?b ?l))
      (not (ontable ?b))
    )
  )
)
