function load_models() {
    let $spinner = $(this).find(".spinner-grow");
    let $button = $(this);
    $spinner.removeClass("d-none");
    $button.prop("disabled", true);

    $.getJSON("/models", function (models) {
        let lista = $("#modelList");
        lista.empty();
        $.each(models, function (index, model) {
            lista.append(new Option(model, model));
        });
        $spinner.addClass("d-none");
        $button.removeAttr("disabled");
        loadGeolets()
        loadTrajectories()
    });
}

function load_selectors() {
    let $spinner = $(this).find(".spinner-grow");
    let $button = $(this);
    $spinner.removeClass("d-none");
    $button.prop("disabled", true);

    $.getJSON("/selectors", function (models) {
        let lista = $("#selectorList");
        lista.empty();
        $.each(models, function (index, model) {
            lista.append(new Option(model, model));
        });
        $spinner.addClass("d-none");
        $button.removeAttr("disabled");
    });
}

function loadGeolets() {
    let $spinner = $(this).find(".spinner-grow");
    let $button = $(this);
    $spinner.removeClass("d-none");
    $button.prop("disabled", true);

    var selectedModel = $("#modelList").val();

    $.getJSON("/geolets", {selected_model: selectedModel}, function (models) {
        let lista = $("#geoletList");
        lista.empty();
        $.each(models, function (index, model) {
            lista.append(new Option(model, model));
        });
        $spinner.addClass("d-none");
        $button.removeAttr("disabled");
    });
}

function loadTrajectories() {
    let $spinner = $(this).find(".spinner-grow");
    let $button = $(this);
    $spinner.removeClass("d-none");
    $button.prop("disabled", true);

    var selectedModel = $("#modelList").val();
    console.log(selectedModel)

    $.getJSON("/trj", {selected_model: selectedModel}, function (models) {
        let lista = $("#trjList");
        lista.empty();
        $.each(models, function (index, model) {
            lista.append(new Option(model, model));
        });
        $spinner.addClass("d-none");
        $button.removeAttr("disabled");
    });
}

function runSelector() {
    let $spinner = $(this).find(".spinner-grow");
    let $button = $(this);
    $spinner.removeClass("d-none");
    $button.prop("disabled", true);

    var selectedModel = $("#modelList").val();
    var selectedSelector = $("#selectorList").val();

    let lista = $("#selectedGeolets");
    lista.empty()
    $("#similarGeolets").empty();

    $.getJSON("/run_selector", {selected_model: selectedModel, selected_selector: selectedSelector}, function (urls) {
        $.each(urls, function (index, data) {
            img_url = data[0]
            img_data = data[1]
            var geolet = img_url.split(/[/ ]+/).pop().replace(".png", "");
            var trj = geolet.split(/[_ ]+/).pop();

            let imgElement = $(`
                    <div class="col-sm-2 mx-2", id="${geolet}">
                        <div class="card">
                            <h6 class="card-header">Geolet: ${geolet}</h6>
                            <img class="card-img-top" src="${img_url}">
                            <div class="card-body">
                                <p class="card-text">Extracted from trajectory: ${trj}</p>
                                <button onclick="set_values('${geolet}', '${trj}')" class="btn btn-primary">
                                    <span class="sr-only">Select</span>
                                </button>
                                <button onclick="find_similar('${geolet}')" class="btn btn-outline-primary">
                                    <span class="spinner-grow spinner-grow-sm  d-none" role="status" aria-hidden="true"></span>
                                    <span class="sr-only">Similar</span>
                                </button>
                            </div>
                        </div>
                    </div>
                    `).data("clu_elements", img_data);
            lista.append(imgElement)
        });
        $spinner.addClass("d-none");
        $button.removeAttr("disabled");
    });
}

function find_similar(geolet){
    let $spinner = $(this).find(".spinner-grow");
    let $button = $(this);
    $spinner.removeClass("d-none");
    $button.prop("disabled", true);

    var clu_elements = $("#"+geolet).data('clu_elements');
    var selectedModel = $("#modelList").val();
    var selectedSelector = $("#selectorList").val();
    console.log(clu_elements);

    let lista = $("#similarGeolets");
    lista.empty()

    $.getJSON("/find_similar",
        {geolet: geolet, clu_elements: clu_elements, selected_model: selectedModel, selected_selector: selectedSelector},
        function (urls) {
        $.each(urls, function (index, data) {
            img_url = data[0]
            img_data = data[1]
            var geolet = img_url.split(/[/ ]+/).pop().replace(".png", "");
            var trj = geolet.split(/[_ ]+/).pop();

            let imgElement = $(`
                    <div class="col-sm-2 mx-2", id="${geolet}">
                        <div class="card">
                            <h6 class="card-header">Geolet: ${geolet}</h6>
                            <img class="card-img-top" src="${img_url}">
                            <div class="card-body">
                                <p class="card-text">Extracted from trajectory: ${trj}</p>
                                <button onclick="set_values('${geolet}', '${trj}')" class="btn btn-primary">
                                    <span class="sr-only">Select</span>
                                </button>
                                <button onclick="find_similar('${geolet}')" class="btn btn-outline-primary">
                                    <span class="spinner-grow spinner-grow-sm  d-none" role="status" aria-hidden="true"></span>
                                    <span class="sr-only">Similar</span>
                                </button>
                            </div>
                        </div>
                    </div>
                    `).data("clu_elements", img_data);
            lista.append(imgElement)
        });
        $spinner.addClass("d-none");
        $button.removeAttr("disabled");
    });
}


function set_values(geolet, trj){
    $("#trjList_input").val(trj);
    $("#geoletList_input").val(geolet);
}

function generate_map(){
    let $spinner = $(this).find(".spinner-grow");
    let $button = $(this);
    $spinner.removeClass("d-none");
    $button.prop("disabled", true);

    var selectedModel = $("#modelList").val();
    var selectedTrj = $("#trjList_input").val();
    var selectedGeolet = $("#geoletList_input").val();

    let lista = $("#map");
    lista.empty()

    $.getJSON("/generate_map", {
        selected_model: selectedModel,
        selected_trj: selectedTrj,
        selected_geolet: selectedGeolet
    }, function (urls) {
        $.each(urls, function (index, map_url) {
            lista.append('<iframe src=\"' + map_url + '\" width=\"100%\" height=\"700px\"></iframe>')
        });
        $spinner.addClass("d-none");
        $button.removeAttr("disabled");
    });
}


$(document).ready(function() {
    load_models()
    load_selectors()

    //========= EVENTS
    $("#loadModels").click(load_models);
    $("#loadSelectors").click(load_selectors);
    $("#runSelector").click(runSelector);
    $("#loadTrajectories").click(loadTrajectories);
    $("#loadGeolets").click(loadGeolets);
    $("#generate_map").click(generate_map);
});