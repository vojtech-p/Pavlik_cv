 function class = apneaDetection(data, fvz)

    if nargin < 2
        fvz = 32;
    end


%% Vykreslení signálů před úpravou:

    %{
    figure
    plot(data.Flow,'Color','b')
    xlabel('Počet vzorků')
    ylabel('Hodnota signálu')
    title('Signály před úpravou: ')
    hold on
    plot(data.Thor, 'Color', 'g')
    hold on
    plot(data.Abdo, 'Color', 'r')
    hold off
    legend('Průtok vzduchu', 'Pohyb hrudníku', 'Pohyb břicha')
    %}


%% Filtrace signálů:

    flow_data = filtrace(data.Flow, 21);
    thor_data = filtrace(data.Thor, 11);
    abdo_data = filtrace(data.Abdo, 11);


%% Normalizace dat:

    flow_data = normalizace(flow_data);
    thor_data = normalizace(thor_data);
    abdo_data = normalizace(abdo_data);


%% Tresholdy:

    treshold_flow = mean(flow_data) * 0.7;
    treshold_thor = mean(thor_data) * 0.4;
    treshold_abdo = mean(abdo_data) * 0.4;


%% Vykreslení signálů po úpravě:

    %{
    figure
    plot(flow_data,'Color','b')
    xlabel('Počet vzorků')
    ylabel('Hodnota signálu')
    title('Signály po úpravě: ')
    hold on
    plot(thor_data, 'Color', 'g')
    hold on
    plot(abdo_data, 'Color', 'r')
    hold off
    legend('Průtok vzduchu', 'Pohyb hrudníku', 'Pohyb břicha')
    %}


%% Detekce aktivity signálů:

    aktivita_flow = aktivita(flow_data, 10, treshold_flow);
    aktivita_thor = aktivita(thor_data, 15, treshold_thor);
    aktivita_abdo = aktivita(abdo_data, 15, treshold_abdo);


%% Vykreslení aktivity signálů:

    %{
    figure
    area(linspace(0,length(flow_data),length(flow_data)),aktivita_flow,'FaceColor','b', 'EdgeColor','b')
    alpha(.999)
    hold on
    plot(flow_data, 'Color', 'b')
    hold on

    area(linspace(0,length(thor_data),length(thor_data)),aktivita_thor * 0.9,'FaceColor','g', 'EdgeColor','g')
    alpha(.5);
    hold on
    plot(thor_data, 'Color', 'g')
    hold on
    
    area(linspace(0,length(abdo_data),length(abdo_data)),aktivita_abdo * 0.8,'FaceColor','r', 'EdgeColor','r')
    alpha(.1);
    hold on
    plot(abdo_data, 'Color', 'r')
    hold on

    legend('', 'Průtok vzduchu', '', 'Pohyb hrudníku', '', 'Pohyb břicha')
    %}
    

%% Vymezení hranic nejdelšího intervalu bez flow:

    start = 0;
    stop = 0;
    i = 1;
    while i < length(flow_data)
        if aktivita_flow(i) == 0
            j = i;
            while j < length(flow_data) && aktivita_flow(j) == 0
                j = j + 1;
            end
            if j - i > stop - start
                stop = j;
                start = i;
            end
            i = j + 1;
        else
            i = i + 1;
        end
    end

    % Počítání pokusů o nádech bez flow:
    i = start;
    thor_count = 0;
    abdo_count = 0;
    while i <= stop - 1
        if (aktivita_thor(i) == 0) && (aktivita_thor(i+1) > 0)
            thor_count = thor_count + 1;
        end
        if (aktivita_abdo(i) == 0) && (aktivita_abdo(i+1) > 0)
            abdo_count = abdo_count + 1;
        end
        i = i + 1;
    end


    %% Detekce hypopnoe:
    
    [pks, locs] = findpeaks(flow_data, MinPeakHeight=treshold_flow);
    hypopnea = false;
    i = 1;
    while i < length(pks)
        j = i + 2;
        while (j < length(pks)) && (pks(j) <= pks(i) * 0.55)
            j = j + 1;
        end
        if j < length(pks)
            if (locs(j) - locs(i) >= fvz * 10)
                hypopnea = true;
            end
        end
        i = i + 1;
    end

    %{
    figure
    plot(flow_data, 'Color', 'b')
    xlabel('Počet vzorků')
    ylabel('Hodnota signálu')
    title('Sledování poklesu peaků signálu: ')
    hold on
    scatter(locs, pks)
    hold off
    legend('Průtok vzduchu', 'Detekované peaky')
    %}


    %% Klasifikace apnoe:

    apnoe = false;
    if stop - start >= fvz * 10
        apnoe = true;
        if thor_count > 4 || abdo_count > 4
            class = 2;
        else
            class = 1;
        end
    end

    if apnoe == false
        if hypopnea
            class = 3;
        else
            class = 4;
        end
    end


 end


%% Použité funkce:

% Funkce pro mediánovou filtraci signálů:
% Vstup - data, délka plovoucího okna med. filtru
% Výstup - d = filtrovaný signál mediánovým filtrem
function d = filtrace(data, okno)
    pulo = (okno-1)/2;
    d = zeros(0, length(data));                               
    for i = (pulo+1) : (length(data) - pulo)
        okno1 = data(i-pulo:i+pulo);
        d(i) = median(okno1);
    end
end

% Funkce pro normalizaci signálů v rozmezí 0-1:
% Vstup - data
% Výstup - n = normalizovaná data s minimem v 0 a maximem v 1
function n = normalizace(data)
    n = data;
    avg = mean(n);
    for i = 1:length(n)
        if data(i) - avg >= 0
            n(i) = data(i) - avg;
        else
            n(i) = 0;
        end
    end
    maximum = max(n);
    for i = 1 : length(n)
        n(i) = n(i) / maximum;
    end
end

% Funkce pro detekci aktivity signálů:
% Vstup - data, délka plovoucího okna, limit determinující aktvitu 0(pod limitem) / 1(nad limitem)
% Výstup - a = binární data o délce vstupního signálu určující aktivitu signálu
function a = aktivita(data, okno, treshold)
    a = zeros(length(data), 1);
    Nokno = floor(length(data) / okno);
    for i = 1 : Nokno
        usek = data((i*okno - okno+1) : (i*okno));
        if mean(usek) >= treshold
            a((i*okno - okno+1) : (i*okno)) = 0.9;
        end
    end
end