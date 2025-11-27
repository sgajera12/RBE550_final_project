;; blocksworld.pddl
;; PDDL Domain for Blocksworld

(define (domain blocksworld)
  (:requirements :strips)
  
  (:predicates
    (on ?x ?y)         ; block x is on block y
    (ontable ?x)       ; block x is on the table
    (clear ?x)         ; nothing is on top of block x
    (holding ?x)       ; robot is holding block x
    (handempty)        ; robot gripper is empty
  )
  
  ;; Pick up a block from the table
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
  
  ;; Put down a held block onto the table
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
  
  ;; Stack a held block on top of another block
  (:action stack
    :parameters (?x ?y)
    :precondition (and
      (holding ?x)
      (clear ?y)
    )
    :effect (and
      (not (holding ?x))
      (not (clear ?y))
      (clear ?x)
      (handempty)
      (on ?x ?y)
    )
  )
  
  ;; Unstack a block from another block
  (:action unstack
    :parameters (?x ?y)
    :precondition (and
      (on ?x ?y)
      (clear ?x)
      (handempty)
    )
    :effect (and
      (holding ?x)
      (clear ?y)
      (not (clear ?x))
      (not (handempty))
      (not (on ?x ?y))
    )
  )
)
