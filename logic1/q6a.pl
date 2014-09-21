central_france :- paris.
cold5 :- paris.
overcast_skies :- paris.
brittany :- nantes.
dry :- brittany.
clear_skies :- brittany.
mild15 :- nantes.
sunny :- nantes.
city_walk :-
 dry, clear_skies, mild15, sunny.
nantes :- fast_train.
fast_train.
paris :- false.

?- city_walk, \+drizzling_rain.
