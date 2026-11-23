-module(llm_engine).
-author("Nilton Aleixo, Adolfo Neto, Maria Claudia Emer").
-export([train/1, generate/4, greedy_sampler/2, make_temp_sampler/2, tokenize/1]).

%% Tipos de Dados
-type token() :: string().
-type model() :: #{token() => [{token(), float()}]}.

%% -------------------------------------------------------------
%% 1. Processamento de Colecoes e Tokenizacao (Collection Proc.)
%% -------------------------------------------------------------
-spec tokenize(string()) -> [token()].
tokenize(Text) ->
    Lower = string:lowercase(Text),
    Words = string:tokens(Lower, " \t\r\n.,;:!?\"()[]{}-"),
    lists:filter(fun(W) -> length(W) > 0 end, Words).

%% -------------------------------------------------------------
%% 2. Treinamento: Construcao do Modelo de N-gramas (Bi-gramas)
%% -------------------------------------------------------------
-spec train([string()]) -> model().
train(CorpusLines) ->
    AllTokens = lists:flatmap(fun tokenize/1, CorpusLines),
    Counts = count_bigrams(AllTokens, #{}),
    normalize_counts(Counts).

%% Recursao em cauda e casamento de padroes para contar pares
count_bigrams([W1, W2 | Rest], Acc) ->
    NextMap = maps:get(W1, Acc, #{}),
    UpdatedInner = maps:update_with(W2, fun(C) -> C + 1 end, 1, NextMap),
    count_bigrams([W2 | Rest], maps:put(W1, UpdatedInner, Acc));
count_bigrams([_Last], Acc) -> Acc;
count_bigrams([], Acc) -> Acc.

%% Normalizacao das frequencias em distribuicoes de probabilidade
normalize_counts(Counts) ->
    maps:map(fun(_Prefix, SuffixMap) ->
        Total = lists:foldl(fun({_, C}, Sum) -> Sum + C end, 0, maps:to_list(SuffixMap)),
        Dist = lists:map(fun({Tok, C}) -> {Tok, C / Total} end, maps:to_list(SuffixMap)),
        lists:sort(fun({_, P1}, {_, P2}) -> P1 >= P2 end, Dist)
    end, Counts).

%% -------------------------------------------------------------
%% 3. Funcoes de Primeira Classe e Amostradores (Samplers)
%% -------------------------------------------------------------
%% Amostrador Ganancioso (Greedy / Determinista, T -> 0)
-spec greedy_sampler(model(), token()) -> token().
greedy_sampler(Model, PrevToken) ->
    case maps:get(PrevToken, Model, not_found) of
        [{BestToken, _} | _] -> BestToken;
        not_found -> "<fim>"
    end.

%% Amostrador Parametrico: Fabricador de Funcao Anonima (Higher-Order)
-spec make_temp_sampler(float(), float()) -> fun((model(), token()) -> token()).
make_temp_sampler(Temperature, TopP) ->
    fun(Model, PrevToken) ->
        case maps:get(PrevToken, Model, not_found) of
            not_found -> "<fim>";
            Dist ->
                Scaled = lists:map(fun({Tok, P}) -> 
                    {Tok, math:pow(P, 1.0 / max(0.01, Temperature))} 
                end, Dist),
                Norm = normalize_distribution(Scaled),
                Filtered = filter_top_p(Norm, TopP),
                Weighted = normalize_distribution(Filtered),
                sample_cdf(to_cdf(Weighted, 0.0), rand:uniform())
        end
    end.

%% Filtragem Top-P (Nucleus Sampling) via foldl
filter_top_p(Dist, TopP) ->
    {Keep, _} = lists:foldl(fun({Tok, P}, {AccList, AccProb}) ->
        if 
            AccProb < TopP -> {[{Tok, P} | AccList], AccProb + P};
            true -> {AccList, AccProb}
        end
    end, {[], 0.0}, Dist),
    lists:reverse(Keep).

%% Normalizacao de lista ponderada
normalize_distribution(Dist) ->
    Total = lists:foldl(fun({_, P}, Acc) -> Acc + P end, 0.0, Dist),
    lists:map(fun({Tok, P}) -> {Tok, P / Total} end, Dist).

%% Conversao da distribuicao discreta em Funcao de Distribuicao Acumulada (CDF)
to_cdf([{Tok, P} | Rest], Acc) ->
    Current = Acc + P,
    [{Tok, Current} | to_cdf(Rest, Current)];
to_cdf([], _Acc) -> [].

%% Amostragem na CDF via casamento de padroes
sample_cdf([{Tok, Limit} | _], Roll) when Roll =< Limit -> Tok;
sample_cdf([_ | Rest], Roll) -> sample_cdf(Rest, Roll);
sample_cdf([], _Roll) -> "<fim>".

%% -------------------------------------------------------------
%% 4. Laco de Geracao Autoregressiva (Recursao em Cauda)
%% -------------------------------------------------------------
-spec generate(model(), [token()], integer(), fun((model(), token()) -> token())) -> [token()].
generate(_Model, Tokens, 0, _Sampler) ->
    lists:reverse(Tokens);
generate(Model, [Prev | _] = Acc, Steps, Sampler) ->
    NextToken = Sampler(Model, Prev),
    case NextToken of
        "<fim>" -> lists:reverse(Acc);
        _ -> generate(Model, [NextToken | Acc], Steps - 1, Sampler)
    end.