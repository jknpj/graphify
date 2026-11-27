// DM parser fixture: strings/comments must be invisible, interpolation is code,
// verbs/macros/comma vars are declarations, and receivers carry types.
/proc/unique_by_name()
	return 1

/obj/feature
	var/obj/feature/part/part
	var/static/uses = 0
	var/first = 1, second = 2

	New()
		..()
		unique_by_name()

	proc/touch(mob/living/M as mob)
		log_me("unique_by_name()")
		M.scream("x")
		src.unique_by_name()
		part.touch(M)
		unicorn("[unique_by_name()]")
		// unique_by_name() in a comment
		return

	verb/use_feature()
		unique_by_name()

/obj/feature/part
	proc/touch(mob/living/M as mob)
		return ..()

/obj/feature/part/special
	proc/touch(mob/living/M as mob)
		return ..()

/datum/listy
	proc/pick_one()
		var/blocked = list(
			/obj/feature/part,
			)
		to_chat(usr, "x")

#define FEATURE_MACRO(x) unique_by_name()

/*
/obj/phantom
	proc/haunt()
		return unique_by_name()
*/
