#Request.py
#import time

class Request(object):

    def __init__(self, matchId=None):
        self.matchId = matchId

    def process(receptionTime):
        '''ABC method to returns the string that results from processing an IP request sent to a GGP player

        :input: receptionTime: 64 bit integer indicatting the time that the request was received
        '''
        return ''

    def  getMatchId(self):
        '''ABC method to returns the string representing the matchID for the match this request is for'''
        return self.matchId

    def toString(self):
        '''String representation of this Request instance'''
        return ''

class AbortRequest(Request):

    def __init__(self, gamer=Gamer(), matchId=''):
        self.gamer = gamer
        self.matchId = matchId

    def getMatchId(self):
       return self.matchId

    def process(receptionTime):
        '''
        First, check to ensure that this abort request is for the match
        we're currently playing. If we're not playing a match, or we're
        playing a different match, send back "busy".
        '''
        if (self.gamer.getMatch() == None || !gamer.getMatch().getMatchId().equals(matchId))
        {
            GamerLogger.logError("GamePlayer", "Got abort message not intended for current game: ignoring.");
            gamer.notifyObservers(new GamerUnrecognizedMatchEvent(matchId));
            return "busy";
        }

        // Mark the match as aborted and notify observers
        gamer.getMatch().markAborted();
        gamer.notifyObservers(new GamerAbortedMatchEvent());
        try {
            gamer.abort();
        } catch (AbortingException e) {
            GamerLogger.logStackTrace("GamePlayer", e);
        }

        // Once the match has ended, set 'roleName' and 'match'
        // to NULL to indicate that we're ready to begin a new match.
        gamer.setRoleName(null);
        gamer.setMatch(null);

        return "aborted";
    }

    @Override
    public String toString()
    {
        return "abort";
    }
}