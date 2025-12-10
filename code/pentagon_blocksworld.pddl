;; pentagon_blocksworld.pddl
;; Blocksworld-style domain with abstract locations for pentagon slots.

(define (domain pentagonworld)
  (:requirements :strips)

  (:predicates
    (on ?x ?y)          ; block x is on block y
    (ontable ?x)        ; block x is on the table
    (clear ?x)          ; nothing on top
    (holding ?x)        ; robot is holding x
    (handempty)         ; gripper is empty

    (at ?x ?l)          ; block x is at abstract location l
    (base-loc ?l)       ; l is a base pentagon slot
    (top-loc ?l)        ; l is a top pentagon slot
  )

  ;; Pick up from table
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

  ;; Put a held block onto the table at some base slot (base pentagon).
  (:action put-down-base
    :parameters (?x ?l)
    :precondition (and
      (holding ?x)
      (base-loc ?l)
    )
    :effect (and
      (not (holding ?x))
      (clear ?x)
      (handempty)
      (ontable ?x)
      (at ?x ?l)
    )
  )

  ;; Put a held block on top of a base block at a top slot (bridged top pentagon).
  (:action put-down-top
    :parameters (?x ?l ?y)
    :precondition (and
      (holding ?x)
      (top-loc ?l)
      (clear ?y)
      (ontable ?y)
    )
    :effect (and
      (not (holding ?x))
      (clear ?x)
      (handempty)
      (on ?x ?y)
      (at ?x ?l)
      (not (clear ?y))
    )
  )
)
