"""The shared Create wizard in node; no browser, backend or model is started."""
from pathlib import Path

import pytest

from test_shell_page import run


PAGE = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
HOME_PAGE = (Path(__file__).resolve().parents[1] / "app" / "static" / "home.js").read_text(encoding="utf-8")
CREATE = (
    HOME_PAGE[HOME_PAGE.index("const startCreate ="):HOME_PAGE.index("// the machine, in one line")]
    + PAGE[PAGE.index("const C ="):PAGE.index("const IMAGE_SIZES")]
    + PAGE[PAGE.index("const IMAGE_SIZES"):PAGE.index("// A number of passes")]
    + PAGE[PAGE.index("const STEPS ="):PAGE.index("/* ---------- Production")]
)
SETUP = """
const makeElement = el;
globalThis.el = (tag, attrs, ...kids) => {
  const node = makeElement(tag, attrs, ...kids);
  node.type = node.attrs.type || "";
  node.listeners = {};
  node.setAttribute = (key, value) => { node.attrs[key] = value; };
  node.addEventListener = (name, handler) => { node.listeners[name] = handler; };
  return node;
};
S.lengths = {fps:24, shortest:22, longest:362, step:17, presets:{small:362}};
S.durations = {5:124}; S.qualities = {standard:"Standard"};
S.presets.small = {size:[864,480],label:"Small"};
S.ref2va = false; S.ref2va_turbo = false; S.sharp = null;
S.windows_vsa = {available:true,installed:true,qualification_enabled:true,
  lengths:Array.from({length:21},(_,i) => { const frames=22+17*i;
    return {frames,seconds:frames/24,installed:true,available:true,qualified:frames===22,scope:"864x480, four passes"};}),
  qualification_summary:{checked_count:1,total_count:21}};
globalThis.clearButton = () => null;
globalThis.keep = node => node; globalThis.encoderOf = () => "";
globalThis.negNote = () => null; globalThis.negShow = async () => {};
globalThis.summary = () => el("div",{},"Supported summary");
globalThis.videoControls = () => el("div",{},"Supported recipe controls");
globalThis.loraWants = () => []; globalThis.sparksPick = () => null;
globalThis.jobCard = job => el("div",{},job.status);
globalThis.LIBF = {kind:"all",q:""};
globalThis.MU = {under:null,videos:null};
"""


def create_run(code):
    return run(SETUP + CREATE + """
      C.videoBackend = "vsa"; C.prompt = "Written exactly."; C.seed = 99;
      VSA.image = "uploads/reference.png"; VSA.frames = 158;
      VSA.reference = {path:VSA.image,width:512,height:288,ready:true,resized:false};
    """ + code)


def production_run(code):
    production = PAGE[PAGE.index("/* ---------- Production"):PAGE.index("/* ---------- Editor")]
    return create_run(production + """
      globalThis.CLEARED=null; globalThis.SPK={n:1};
      globalThis.bind=(o,k,n)=>n; globalThis.select=()=>el("select");
      globalThis.loraHub=()=>null; globalThis.lengthPicker=()=>null;
      answers["/api/projects"]=[];
    """ + code)


@pytest.mark.parametrize("status,finished,retry,busy", [
    ("queued", False, False, True), ("running", False, False, True),
    ("failed", False, True, False), ("cancelled", False, True, False),
    ("done", True, False, False),
])
def test_production_board_only_delivers_finished_segments_and_retries_cancelled(status, finished, retry, busy):
    out = production_run("""
      project={id:"test",segments:[{start:"vsa",job:"one",output:"renders/one/video.mp4",seconds:3,frames:73}]};
      jobs=[{id:"one",status:""" + repr(status) + """}];
      const root=viewProduction(); await new Promise(resolve=>setImmediate(resolve));
      out.editorDisabled=button(root,"Send rendered segments").attrs.disabled;
      out.batchDisabled=button(root,"Generate every segment").attrs.disabled;
      out.renderDisabled=button(root,"Render again").attrs.disabled;
      out.exactOverview=text(root).includes("3.042 s total") && text(root).includes("73 frames");
    """)
    assert out == {"editorDisabled": not finished, "batchDisabled": not retry, "renderDisabled": busy, "exactOverview": True}


def test_production_board_blocks_duplicate_submission_before_and_after_queue_reply():
    out = production_run("""
      project={id:"test",preset:"small",segments:[{start:"vsa",vsaImage:"uploads/ref.png",prompt:"Reference."}]};
      const savedApi=api; let release; const reply=new Promise(resolve=>{release=resolve;});
      globalThis.api=async(path,body)=>{if(path==="/api/jobs"){calls.push([path,body]);return await reply;}return savedApi(path,body);};
      const first=generateSeg(0);
      try {await generateSeg(0);} catch(e) {out.pendingError=e.message;}
      release({id:"one",status:"queued"}); await first;
      try {await generateSeg(0);} catch(e) {out.queuedError=e.message;}
      out.count=calls.filter(c=>c[0]==="/api/jobs").length;
    """)
    assert out["pendingError"] and out["queuedError"] and out["count"] == 1


def test_production_board_submission_failure_allows_retry():
    out = production_run("""
      project={id:"test",preset:"small",segments:[{start:"vsa",vsaImage:"uploads/ref.png",prompt:"Reference."}]};
      answers["/api/jobs"]=new Error("Admission refused");
      try {await generateSeg(0);} catch(e) {out.error=e.message;}
      out.busy=segBusy(project.segments[0]);
      answers["/api/jobs"]={id:"retry",status:"queued"}; await generateSeg(0);
      out.job=project.segments[0].job;
    """)
    assert out == {"error": "Admission refused", "busy": False, "job": "retry"}


@pytest.mark.parametrize("start", ["text", "image", "previous"])
def test_production_board_keeps_supported_h3_routes(start):
    out = production_run("""
      project={id:"test",preset:"small",quality:"standard",segments:[
        {output:"renders/previous.mp4"},
        {start:""" + repr(start) + """,image:"uploads/first.png",prompt:"Shot.",seconds:5,seed:7}]};
      answers["/api/jobs"]={id:"supported",status:"queued"};
      await generateSeg(1);
      out.job=calls.find(c=>c[0]==="/api/jobs")[1];
      out.busy=segBusy(project.segments[1]);
    """)
    assert out["job"]["kind"] == "video" and out["busy"]
    params = out["job"]["params"]
    assert params["prompt"] == "Shot." and params["preset"] == "small" and params["seed"] == 7
    assert ("image" in params) == (start == "image")
    assert ("first_frame_from" in params) == (start == "previous")
    if start == "previous":
        assert params["first_frame_from"] == "renders/previous.mp4"


def test_production_board_editor_handoff_uses_actual_saved_duration():
    out = production_run("""
      project={id:"test",timeline:[],segments:[
        {job:"done",output:"renders/done/video.mp4",seconds:3,frames:73},
        {job:"running",output:"renders/running/video.mp4",seconds:5}]};
      jobs=[{id:"done",status:"done"},{id:"running",status:"running"}];
      answers["/api/probe?path=renders%2Fdone%2Fvideo.mp4"]={duration:73/24};
      const root=viewProduction(); await new Promise(resolve=>setImmediate(resolve));
      await button(root,"Send rendered segments").attrs.onclick();
      out.timeline=project.timeline;out.tab=tab;
      out.saved=calls.find(c=>c[0]==="/api/projects/test")[1].timeline;
    """)
    expected = [{"path": "renders/done/video.mp4", "in": 0, "out": 0, "duration": 73 / 24}]
    assert out["timeline"] == out["saved"] == expected
    assert out["tab"] == "Editor"


@pytest.mark.parametrize("failure", ["probe", "duration", "save"])
def test_editor_handoff_failure_preserves_existing_timeline(failure):
    out = production_run("""
      project={id:"test",timeline:[{path:"kept.mp4"}],segments:[]};
      answers["/api/probe?path=good.mp4"]={duration:7.292};
      answers["/api/probe?path=bad.mp4"]=""" + (
        'new Error("Probe refused")' if failure == "probe" else '{duration:0}' if failure == "duration" else '{duration:5}'
    ) + """;
      """ + ('answers["/api/projects/test"]=new Error("Save refused");' if failure == "save" else "") + """
      try {await replaceTimeline(["good.mp4","bad.mp4"]);} catch(e) {out.error=e.message;}
      out.timeline=project.timeline;out.tab=tab;
      out.writes=calls.filter(c=>c[0]==="/api/projects/test").length;
    """)
    assert out["error"] and out["timeline"] == [{"path": "kept.mp4"}]
    assert out["tab"] != "Editor"
    assert out["writes"] == (1 if failure == "save" else 0)


def test_production_board_routes_explicit_vsa_segments_without_standard_settings():
    production = PAGE[PAGE.index("/* ---------- Production"):PAGE.index("/* ---------- Editor")]
    out = create_run(production + """
      globalThis.project={id:"test",preset:"small",quality:"standard",segments:[{start:"vsa",vsaImage:"uploads/ref.png",seconds:5,seed:0,scene:"A baker smiles.",prompt:""}]};
      answers["/api/compose"]={prompt:"Reference <Picture 1>."};
      answers["/api/jobs"]={id:"board-vsa"};
      await generateSeg(0);
      out.compose=calls.find(c=>c[0]==="/api/compose")[1];out.job=calls.find(c=>c[0]==="/api/jobs")[1];
      out.output=project.segments[0].output;
    """)
    assert out["compose"]["referenced"] and out["compose"]["references"] == 1 and not out["compose"]["first_frame"]
    assert out["job"]["kind"] == "video-vsa"
    assert out["job"]["params"] == {"image": "uploads/ref.png", "prompt": "Reference <Picture 1>.", "frames": 124, "seed": 0}
    assert out["output"] == "renders/board-vsa-vsa/video.mp4"


@pytest.mark.parametrize("reason", ["lora", "canvas"])
def test_production_board_refuses_unsupported_vsa_settings(reason):
    production = PAGE[PAGE.index("/* ---------- Production"):PAGE.index("/* ---------- Editor")]
    out = create_run(production + """
      globalThis.project={id:"test",preset:"small",segments:[{start:"vsa",vsaImage:"uploads/ref.png",seconds:5,seed:7,prompt:"Reference."}]};
      globalThis.lorasOf=()=>""" + ('[{id:"style"}]' if reason == "lora" else "[]") + """;
      """ + ('S.presets.small.size=[1280,736];' if reason == "canvas" else "") + """
      try { await generateSeg(0); } catch(e) {out.error=e.message;}
      out.queued=calls.some(c=>c[0]==="/api/jobs");
    """)
    assert out["error"] and not out["queued"]


def test_vsa_uses_all_five_shared_steps_without_supported_recipe_controls():
    out = create_run("""
      out.pages = [];
      for (let step=0; step<5; step++) {
        C.step = step;
        const root = viewCreate();
        const tabs = find(root,n=>n.attrs.class==="steps");
        out.pages.push({text:text(root),steps:tabs.kids.length});
      }
    """)
    assert all(page["steps"] == 5 for page in out["pages"])
    shot, picture, sound, review, render = [page["text"] for page in out["pages"]]
    assert "Describe the scene" in shot and "What happens to the characters" in shot and "Camera (optional)" in shot
    assert "Supported recipe controls" not in shot and "The clip starts from" not in shot
    assert "four passes" in shot and "not a guaranteed first frame" in shot
    assert "Upload a reference picture" in picture and "at most 512x288" in picture
    assert "Add videos" not in picture and "last frame" not in picture and "Make scouts in Image" in picture
    assert "Add a line" in sound and "Soundscape" in sound and "Music (N/A for none)" in sound
    assert "Review the prompt" in review and "Written exactly." not in review
    assert "Render 158 frames (6.583 s)" in render and "Add to the editor" not in render


def test_vsa_labels_are_not_experimental_in_any_create_step():
    out = create_run("""
      out.pages=[];
      for(let step=0;step<5;step++) { C.step=step; out.pages.push(text(viewCreate())); }
    """)
    assert len(out["pages"]) == 5
    assert all("Ref2VA VSA" in page and "experimental" not in page.lower() for page in out["pages"])


def test_switching_backends_keeps_shared_authoring_and_separate_recipe_inputs():
    out = create_run("""
      Object.assign(C,{scene:"Scene",action:"Action",camera:"Camera",dialogue:[{text:"Hello"}],
        soundscape:"Wind",music:"Piano",image:"uploads/first.png",lastImage:"uploads/last.png",
        start:"own",steps:12,refs:["uploads/other.png"],refVideos:["uploads/source.mp4"],
        refAudios:["uploads/source.wav"],loras:[{name:"look"}],retake:true});
      const kept = () => JSON.stringify({scene:C.scene,action:C.action,camera:C.camera,dialogue:C.dialogue,
        soundscape:C.soundscape,music:C.music,prompt:C.prompt,seed:C.seed});
      const shared = kept();
      await button(viewCreate(),"Supported H3").attrs.onclick();
      const standard = viewCreate();
      out.standard = text(standard);
      out.settings = [C.start,C.image,C.lastImage,C.steps,C.refs,C.refVideos,C.refAudios,C.loras,C.retake];
      await button(standard,"Ref2VA VSA").attrs.onclick();
      out.shared = kept()===shared;
      out.vsa = [VSA.image,VSA.frames,C.videoBackend];
    """)
    assert out["shared"]
    assert "Supported recipe controls" in out["standard"] and "Render 158" not in out["standard"]
    assert out["settings"] == [
        "own", "uploads/first.png", "uploads/last.png", 12, ["uploads/other.png"],
        ["uploads/source.mp4"], ["uploads/source.wav"], [{"name": "look"}], True,
    ]
    assert out["vsa"] == ["uploads/reference.png", 158, "vsa"]


def test_shared_words_and_sound_fields_edit_the_same_create_state():
    out = create_run("""
      C.step=2;
      await button(viewCreate(),"+ Add a line").attrs.onclick();
      let root=viewCreate();
      const line=find(root,n=>n.tag==="textarea" && n.value==="");
      line.value="Hello from the picture."; line.listeners.input();
      const sound=find(root,n=>n.tag==="textarea" && n.attrs.placeholder && n.attrs.placeholder.includes("bakery room"));
      sound.value="Soft wind."; sound.listeners.input();
      out.dialogue=C.dialogue; out.sound=C.soundscape;
      C.step=3; root=viewCreate();
      const prompt=find(root,n=>n.tag==="textarea");
      prompt.value="My exact prompt."; prompt.listeners.input();
      out.prompt=C.prompt;
      C.step=0; root=viewCreate();
      const seed=find(root,n=>n.tag==="input" && n.attrs.max===4294967295);
      seed.type="number"; seed.value="123"; seed.listeners.input();
      out.seed=C.seed;
    """)
    assert out["dialogue"][0]["text"] == "Hello from the picture."
    assert out["sound"] == "Soft wind." and out["prompt"] == "My exact prompt."
    assert out["seed"] == 123


def test_vsa_prompt_reuses_composer_without_leaking_supported_h3_inputs():
    out = create_run("""
      Object.assign(C,{step:3,scene:"Bakery",action:"Waves",camera:"Still",soundscape:"Wind",music:"N/A",
        dialogue:[{text:"Hello",voice:"Calm",subject:3}],start:"refs",refs:["a","b","c"],
        subjects:["Person A","Person B"],refVideos:["source.mp4"],video_subjects:["Person C"],
        refAudios:["source.wav"],audio_notes:["Voice"],video_sound:true,motion:{pack:"old"},lastImage:"last.png"});
      answers["/api/compose"]={prompt:"Composed reference prompt."};
      await button(viewCreate(),"Rebuild from steps").attrs.onclick();
      out.request=calls.find(([path])=>path==="/api/compose")[1]; out.prompt=C.prompt;
    """)
    assert out["request"] == {
        "fields": {
            "scene": "Bakery", "action": "Waves", "camera": "Still", "soundscape": "Wind", "music": "N/A",
            "dialogue": [{"text": "Hello", "voice": "Calm", "subject": 1}],
        },
        "first_frame": False, "last_frame": False, "frames": 158, "seconds": 158 / 24,
        "references": 1, "referenced": True, "loras": [],
    }
    assert out["prompt"] == "Composed reference prompt."


def test_compose_failure_is_visible_and_does_not_replace_reviewed_prompt():
    out = create_run("""
      C.step=3; answers["/api/compose"]=new Error("Composer unavailable");
      await button(viewCreate(),"Rebuild from steps").attrs.onclick();
      out.prompt=C.prompt; out.errors=toasts;
    """)
    assert out["prompt"] == "Written exactly." and out["errors"] == ["Composer unavailable"]


@pytest.mark.parametrize("sequence", [False, True])
def test_vsa_render_sends_only_its_existing_recipe_parameters(sequence):
    out = create_run("""
      C.step=4; C.lastImage="last.png"; C.refVideos=["old.mp4"]; C.steps=20; C.retake=true;
      VSA.mode=""" + ('"sequence"' if sequence else '"single"') + """;
      VSA.sequenceSeconds=8;
      answers["/api/jobs"]={id:"new-job"};
      const root=viewCreate();
      const queue=button(root,VSA.mode==="sequence"?"Render 8-second sequence":"Render 158 frames");
      out.enabled=!queue.attrs.disabled; await queue.attrs.onclick();
      out.request=calls.find(([path])=>path==="/api/jobs")[1]; out.job=VSA.job;
    """)
    params = {"image": "uploads/reference.png", "prompt": "Written exactly.", "seed": 99, "frames": 158}
    if sequence:
        params["sequence_seconds"] = 8
    assert out["enabled"] and out["job"] == "new-job"
    assert out["request"] == {
        "kind": "video-vsa", "title": "VSA sequence" if sequence else "Ref2VA VSA",
        "params": params,
    }


@pytest.mark.parametrize("missing", ["baseline", "picture", "prompt", "duration"])
def test_vsa_render_keeps_existing_blockers(missing):
    out = create_run("""
      C.step=4;
      const missing=""" + repr(missing) + """;
      if(missing==="baseline") { S.windows_vsa.why="Basic check required"; S.windows_vsa.lengths.forEach(row=>row.available=false); }
      if(missing==="picture") VSA.image="";
      if(missing==="prompt") C.prompt="";
      if(missing==="duration") { VSA.mode="sequence"; VSA.sequenceSeconds=61; }
      const root=viewCreate();
      out.disabled=button(root,missing==="duration"?"Render 61-second":"Render 158 frames").attrs.disabled;
      out.text=text(root); out.calls=calls.length;
    """)
    assert out["disabled"] and out["calls"] == 0
    assert {
        "baseline": "Basic check required", "picture": "Choose and check a reference picture in step 2",
        "prompt": "Open step 4", "duration": "whole number of seconds from 1 to 60",
    }[missing] in out["text"]


def test_supported_h3_composition_and_render_stay_on_the_original_routes():
    out = create_run("""
      C.videoBackend="standard"; C.start="own"; C.image="first.png"; C.lastImage="last.png"; C.step=3;
      answers["/api/compose"]={prompt:"Supported prompt."};
      await button(viewCreate(),"Rebuild from steps").attrs.onclick();
      out.compose=calls.find(([path])=>path==="/api/compose")[1];
      C.step=4; answers["/api/jobs"]={id:"supported-job"};
      await button(viewCreate(),"Queue this clip").attrs.onclick();
      out.request=calls.find(([path])=>path==="/api/jobs")[1]; out.job=C.job; out.vsaJob=VSA.job;
    """)
    assert out["compose"]["first_frame"] and out["compose"]["last_frame"]
    assert not out["compose"]["referenced"] and out["compose"]["references"] == 0
    assert out["request"]["kind"] == "video"
    assert out["request"]["params"]["image"] == "first.png"
    assert out["request"]["params"]["last_image"] == "last.png"
    assert out["job"] == "supported-job" and out["vsaJob"] is None


def test_image_creation_is_not_changed_by_the_selected_video_backend():
    out = create_run("""
      C.goes="image"; C.step=0;
      out.text=text(viewCreate()); out.mode=C.mode; out.active=isVsa();
    """)
    assert out["mode"] == "image" and not out["active"]
    assert "The picture" in out["text"] and "Make image" in out["text"]
    assert "Ref2VA VSA" not in out["text"]


def test_scout_handoff_preserves_backend_and_keeps_supported_frame_separate():
    out = create_run("""
      C.image="uploads/supported-first.png";
      startCreate({goes:"image",count:6});
      out.imageBackend=C.videoBackend;
      answers["/api/video-vsa/reference"]={path:"scouts/chosen.png",width:864,height:480,ready:false,resized:false};
      await useVideoPicture("scouts/chosen.png");
      out.state=[C.goes,C.step,C.videoBackend,C.image,VSA.image];
      out.request=calls.find(([path])=>path==="/api/video-vsa/reference")[1];
      out.ready=vsaReferenceReady();
    """)
    assert out["imageBackend"] == "vsa"
    assert out["state"] == ["video", 1, "vsa", "uploads/supported-first.png", "scouts/chosen.png"]
    assert out["request"] == {"path": "scouts/chosen.png"} and not out["ready"]


def test_supported_picture_handoff_stays_a_first_frame_without_vsa_requests():
    out = create_run("""
      C.videoBackend="standard";
      useVideoPicture("scouts/chosen.png");
      out.state=[C.goes,C.start,C.step,C.videoBackend,C.image,VSA.image];
      out.calls=calls;
    """)
    assert out["state"] == ["video", "own", 1, "standard", "scouts/chosen.png", "uploads/reference.png"]
    assert out["calls"] == []


def test_library_picker_offers_stills_and_scouts_but_not_videos():
    out = create_run("""
      VSA.library=true;
      answers["/api/library"]=[{path:"scouts/candidate.png",kind:"image"},{path:"uploads/photo.jpg",kind:"image"},
        {path:"renders/clip.mp4",kind:"video"}];
      answers["/api/video-vsa/reference"]={path:"scouts/candidate.png",width:512,height:288,ready:true,resized:false};
      const root=vsaReference(); await Promise.resolve();
      out.text=text(root); await button(root,"candidate.png").attrs.onclick();
      out.selected=VSA.image; out.closed=!VSA.library; out.ready=vsaReferenceReady();
    """)
    assert "candidate.png" in out["text"] and "photo.jpg" in out["text"] and "clip.mp4" not in out["text"]
    assert out["selected"] == "scouts/candidate.png" and out["closed"] and out["ready"]


def test_oversized_reference_requires_an_explicit_copy_and_uses_returned_path():
    out = create_run("""
      answers["/api/video-vsa/reference"]={path:"scouts/large.png",width:1920,height:1080,ready:false,resized:false};
      await vsaSelectImage("scouts/large.png");
      out.before=[vsaReferenceReady(),calls.length];
      C.step=4; out.blocked=button(viewCreate(),"Render 158 frames").attrs.disabled;
      answers["/api/video-vsa/reference"]={path:"uploads/small-copy.png",source:"scouts/large.png",
        width:512,height:288,ready:true,resized:true};
      await button(vsaReference(),"Create resized reference copy").attrs.onclick();
      out.calls=calls.filter(([path])=>path==="/api/video-vsa/reference");
      out.selected=VSA.image; out.ready=vsaReferenceReady(); out.note=text(vsaReference());
    """)
    assert out["before"] == [False, 1] and out["blocked"]
    assert [body for _, body in out["calls"]] == [
        {"path": "scouts/large.png"}, {"path": "scouts/large.png", "resize": True},
    ]
    assert out["selected"] == "uploads/small-copy.png" and out["ready"]
    assert "original is unchanged" in out["note"]


def test_reference_check_failure_is_visible_and_blocks_rendering():
    out = create_run("""
      answers["/api/video-vsa/reference"]=new Error("Picture cannot be read");
      await vsaSelectImage("uploads/bad.png"); C.step=4;
      out.ready=vsaReferenceReady(); out.disabled=button(viewCreate(),"Render 158 frames").attrs.disabled;
      out.error=text(vsaReference()); out.toasts=toasts; out.busy=VSA.referenceBusy;
    """)
    assert not out["ready"] and out["disabled"] and not out["busy"]
    assert "Picture cannot be read" in out["error"] and out["toasts"] == ["Picture cannot be read"]


def test_a_late_reference_check_cannot_replace_the_newer_picture():
    out = create_run("""
      const pending={};
      globalThis.api=async (path,body)=>new Promise(resolve=>{pending[body.path]=resolve;});
      const first=vsaSelectImage("uploads/first.png");
      const second=vsaSelectImage("uploads/second.png");
      pending["uploads/second.png"]({path:"uploads/second.png",width:512,height:288,ready:true});
      await second;
      pending["uploads/first.png"]({path:"uploads/first.png",width:1920,height:1080,ready:false});
      await first;
      out.image=VSA.image; out.ready=vsaReferenceReady(); out.busy=VSA.referenceBusy;
    """)
    assert out["image"] == "uploads/second.png" and out["ready"] and not out["busy"]


def test_a_late_upload_cannot_replace_a_newer_library_selection():
    out = create_run("""
      let uploadDone;
      globalThis.fetch=()=>new Promise(resolve=>{uploadDone=resolve;});
      const root=vsaReference(), input=find(root,n=>n.tag==="input" && n.attrs.type==="file");
      input.files=["picture"];
      const upload=input.attrs.onchange();
      answers["/api/video-vsa/reference"]={path:"scouts/newer.png",width:512,height:288,ready:true};
      await vsaSelectImage("scouts/newer.png");
      uploadDone({ok:true,json:async()=>({path:"uploads/older.png"})});
      await upload;
      out.image=VSA.image; out.ready=vsaReferenceReady(); out.calls=calls;
    """)
    assert out["image"] == "scouts/newer.png" and out["ready"]
    assert out["calls"] == [["/api/video-vsa/reference", {"path": "scouts/newer.png"}]]


def test_invalid_reference_details_fail_closed():
    out = create_run("""
      answers["/api/video-vsa/reference"]={path:"uploads/invalid.png",width:1920,height:1080,ready:true};
      await vsaSelectImage("uploads/invalid.png");
      out.ready=vsaReferenceReady(); out.error=VSA.referenceError;
    """)
    assert not out["ready"] and "invalid picture details" in out["error"]


def test_explicit_anchor_action_stays_on_supported_h3_but_reference_can_use_vsa():
    actions = (Path(__file__).resolve().parents[1] / "app" / "static" / "actions.js").read_text(encoding="utf-8")
    use = actions[actions.index("const actUse ="):actions.index("function actPanel()")]
    out = create_run(use + """
      answers["/api/video-vsa/reference"]={path:"scouts/turnaround.png",width:512,height:288,ready:true};
      await actUse("scouts/turnaround.png","ref")();
      out.reference=[C.videoBackend,VSA.image];
      await actUse("scouts/first.png","first")();
      out.anchor=[C.videoBackend,C.image,C.start];
    """)
    assert out["reference"] == ["vsa", "scouts/turnaround.png"]
    assert out["anchor"] == ["standard", "scouts/first.png", "own"]


def test_vsa_results_share_editor_download_library_and_soundtrack_actions():
    out = create_run("""
      VSA.job="finished";
      jobs=[{id:"finished",status:"done",outputs:["renders/sequence/video.mp4","renders/sequence/sequence.json"]}];
      globalThis.addToTimeline=path=>calls.push(["timeline",path]);
      const root=viewVsa();
      await button(root,"Add to the editor timeline").attrs.onclick();
      await button(root,"Lay a song under it").attrs.onclick();
      out.soundtrack=MU.under;
      await button(root,"Open in Library").attrs.onclick();
      out.library={...LIBF}; out.calls=calls;
      const links=[];
      const visit=node=>{if(!node||typeof node!=="object")return;
        if(Array.isArray(node)){node.forEach(visit);return;}
        if(node.tag==="a")links.push(node.attrs.href); visit(node.kids);};
      visit(root); out.links=links;
    """)
    assert out["soundtrack"] == "renders/sequence/video.mp4"
    assert out["library"] == {"kind": "video", "q": "renders/sequence/video.mp4"}
    assert out["calls"] == [["timeline", "renders/sequence/video.mp4"]]
    assert out["links"] == ["/files/renders/sequence/video.mp4", "/files/renders/sequence/sequence.json"]
