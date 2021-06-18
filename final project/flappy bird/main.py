import sys
import pygame
import random


class Bird(pygame.sprite.Sprite):
    def __init__(self, position):
        pygame.sprite.Sprite.__init__(self)
        self.rect = pygame.Rect(*position, BIRD_WIDTH, BIRD_HEIGHT)
        self.down_speed = 0  # 重力大小，即每次向下掉落多少
        self.up_speed = 10  # 每次跳跃高度
        self.is_flapped = False  # 跳跃状态，默认不跳跃
        self.is_dead = False  # 死亡状态，默认活着
        self.time_pass = FPS / 1000

    def update(self):
        if self.is_flapped:
            # 小鸟跳跃
            self.down_speed = 0
            self.up_speed -= 30 * self.time_pass  # 速度递减，上升越来越慢
            self.rect.top -= self.up_speed  # 鸟Y轴坐标减小，小鸟上升
            if self.up_speed <= 0:
                self.is_flapped = False
                self.up_speed = 10
                self.down_speed = 0
        else:
            # 小鸟坠落
            self.down_speed += 30 * self.time_pass
            self.rect.bottom += self.down_speed

        if self.rect.top <= 0:
            self.up_speed = 0
            self.down_speed = 0
            self.is_dead = True
        if self.rect.bottom >= BASE_HEIGHT:
            self.up_speed = 0
            self.down_speed = 0
            self.rect.bottom = BASE_HEIGHT
            self.is_dead = True

        return self.is_dead

    def up(self):
        if self.is_flapped:
            self.upspeed = max(12, self.up_speed + 2)
        else:
            self.is_flapped = True

    def draw(self, screen):
        self.img = pygame.image.load('./sources/bird/bird0_0.png')
        screen.blit(self.img, (self.rect.left, self.rect.top))


class Pipe(pygame.sprite.Sprite):
    def __init__(self, position):
        pygame.sprite.Sprite.__init__(self)
        self.left, self.top = position
        pipe_height = PIPE_HEIGHT
        if (self.top + PIPE_HEIGHT) > BASE_HEIGHT:
            pipe_height = BASE_HEIGHT - self.top + 1
        self.rect = pygame.Rect(self.left, self.top, PIPE_WIDTH, pipe_height)
        self.used_for_score = False

    @staticmethod
    def generate_pipe_position():
        # 生成上下两个管道的坐标
        top = int(BASE_HEIGHT * 0.2) + random.randrange(0, int(BASE_HEIGHT * 0.6 - PIPE_GAP))
        return {
            'top': (SCREEN_WIDTH + 25, top - PIPE_HEIGHT),
            'bottom': (SCREEN_WIDTH + 25, top + PIPE_GAP)
        }

    def draw(self, screen):
        self.pipeUp = pygame.image.load('./sources/pipe/pipe_up.png')
        self.pipeDown = pygame.image.load('./sources/pipe/pipe_down.png')
        if (self.top + PIPE_HEIGHT) > BASE_HEIGHT:
            self.pipeUp = pygame.transform.smoothscale(self.pipeUp, (PIPE_WIDTH, BASE_HEIGHT - self.top + 1))
            screen.blit(self.pipeUp, (self.rect.left, self.rect.top))
        else:
            screen.blit(self.pipeDown, (self.rect.left, self.rect.top))


def init_game():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Flappy Bird')
    return screen


def init_sprite():
    # 初始化小鸟类
    bird_position = [SCREEN_WIDTH * 0.2, (BASE_HEIGHT - BIRD_HEIGHT) / 2]
    bird = Bird(bird_position)

    # 初始化管道类
    pipe_sprites = pygame.sprite.Group()
    for i in range(2):
        pipe_pos = Pipe.generate_pipe_position()
        # 添加上方的管道
        pipe_sprites.add(
            Pipe((SCREEN_WIDTH + i * SCREEN_WIDTH / 2,
                  pipe_pos.get('top')[-1])))
        # 添加下方的管道
        pipe_sprites.add(
            Pipe((SCREEN_WIDTH + i * SCREEN_WIDTH / 2,
                  pipe_pos.get('bottom')[-1])))
    return bird, pipe_sprites


def collision(bird, pipe_sprites):
    # 检测管道碰撞
    is_collision = False
    for pipe in pipe_sprites:
        if pygame.sprite.collide_rect(pipe, bird):
            is_collision = True

    # 检测上下边沿碰撞
    is_dead = bird.update()
    if is_dead:
        is_collision = True

    return is_collision


def move_pipe(bird, pipe_sprites, is_add_pipe, score):
    flag = False  # 下一次是否要增加新的pipe的标志位
    for pipe in pipe_sprites:
        pipe.rect.left -= PIPE_SPEED
        # 小鸟飞过pipe 加分
        if pipe.rect.centerx < bird.rect.centerx and not pipe.used_for_score:
            pipe.used_for_score = True
            score += 0.5
        # 增加新的pipe
        if pipe.rect.left < 10 and pipe.rect.left > 0 and is_add_pipe:
            pipe_pos = Pipe.generate_pipe_position()
            pipe_sprites.add(Pipe(position=pipe_pos.get('top')))
            pipe_sprites.add(Pipe(position=pipe_pos.get('bottom')))
            is_add_pipe = False
        # 删除已不在屏幕的pipe, 更新标志位
        elif pipe.rect.right < 0:
            pipe_sprites.remove(pipe)
            flag = True
    if flag:
        is_add_pipe = True
    return is_add_pipe, score


def DrawBackground(screen):
    # 显示背景图和地面
    screen.blit(day, (0, 0))
    screen.blit(land, (0, BASE_HEIGHT))


def DrawStartMenu(screen):
    # 显示标题
    screen.blit(title, ((SCREEN_WIDTH - 178) / 2, 80))
    # 显示引导按钮
    screen.blit(tutorial, (SCREEN_WIDTH * 0.2 - 33, (BASE_HEIGHT - BIRD_HEIGHT) / 2 + BIRD_HEIGHT + 1))


def DrawScore(screen, score, score_img):
    left = SCREEN_WIDTH / 2 - 12
    if (score == 0):
        screen.blit(score_img[score], (left, BASE_HEIGHT / 4))
    while (score != 0):
        score = int(score)
        temp = score % 10
        screen.blit(score_img[temp], (left, BASE_HEIGHT / 4))
        left = left - 24
        score = int(score / 10)


def Gameover(screen, score, best, result_img):
    # 显示gameover界面和计分板
    screen.blit(text_game_over, ((SCREEN_WIDTH - 210) / 2, BASE_HEIGHT / 3))
    screen.blit(score_panel, ((SCREEN_WIDTH - 238) / 2, BASE_HEIGHT / 3 + 60))

    # 显示分数
    left = SCREEN_WIDTH / 2 + 70
    if (score == 0):
        screen.blit(result_img[score], (left, BASE_HEIGHT / 2 + 25))
    while (score != 0):
        score = int(score)
        temp = score % 10
        screen.blit(result_img[temp], (left, BASE_HEIGHT / 2 + 25))
        left = left - 24
        score = int(score / 10)

    left = SCREEN_WIDTH / 2 + 70
    if (best == 0):
        screen.blit(result_img[best], (left, BASE_HEIGHT / 2 + 65))
    while (best != 0):
        best = int(best)
        temp = best % 10
        screen.blit(result_img[temp], (left, BASE_HEIGHT / 2 + 65))
        left = left - 24
        best = int(best / 10)

    # 显示奖牌
    if score > 100:
        screen.blit(gold, ((SCREEN_WIDTH - 180) / 2, BASE_HEIGHT / 3 + 105))
    elif score > 25:
        screen.blit(silver, ((SCREEN_WIDTH - 180) / 2, BASE_HEIGHT / 3 + 105))
    else:
        screen.blit(corper, ((SCREEN_WIDTH - 180) / 2, BASE_HEIGHT / 3 + 105))


def press(is_game_running, bird):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # 点击关闭按钮退出
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if is_game_running:
                bird.up()
            else:  # 游戏结束时回车键继续
                return True


def Main():
    # 是否需要增加管道
    is_add_pipe = True
    # 分数
    score = 0
    global best
    # 是否结束
    is_game_running = True
    # 是否开始
    is_start = False

    # 初始化
    screen = init_game()
    bird, pipe_sprites = init_sprite()
    clock = pygame.time.Clock()

    while True:
        clock.tick(60)
        if is_start == False:
            DrawBackground(screen)
            DrawStartMenu(screen)
            bird.draw(screen)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    is_start = True
        else:
            restart = press(is_game_running, bird)
            if restart:
                return

            DrawBackground(screen)
            DrawScore(screen, score, score_img)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.KEYDOWN:
                    bird.up()

            is_collision = collision(bird, pipe_sprites)

            # 画管道
            for pipe in pipe_sprites:
                pipe.draw(screen)

            bird.draw(screen)  # 画鸟

            if is_collision:
                is_game_running = False  # 如果碰撞 游戏结束
                best = max(best, score)
            if is_game_running:
                is_add_pipe, score = move_pipe(bird, pipe_sprites, is_add_pipe, score)  # 不碰撞 移动管道
            else:
                Gameover(screen, score, best, result_img)
        # 更新画布
        pygame.display.update()
        clock.tick(FPS)


if __name__ == "__main__":

    FPS = 60
    # 屏幕大小
    SCREEN_WIDTH = 288
    SCREEN_HEIGHT = 512
    # 鸟大小
    BIRD_WIDTH = 45
    BIRD_HEIGHT = 45
    # 管道大小
    PIPE_WIDTH = 50
    PIPE_HEIGHT = 300
    # 管道之间间隔
    PIPE_GAP = 150
    # 管道移动速度
    PIPE_SPEED = 4
    # 地面高度
    FLOOR_HEIGHT = 96
    # 游戏有效高度
    BASE_HEIGHT = SCREEN_HEIGHT - FLOOR_HEIGHT

    day = pygame.image.load('./sources/background/day.png')
    land = pygame.image.load('./sources/menu/land.png')
    title = pygame.image.load('./sources/menu/title.png')
    tutorial = pygame.image.load('./sources/menu/tutorial.png')
    text_game_over = pygame.image.load('./sources/gameover/text_game_over.png')
    score_panel = pygame.image.load('./sources/gameover/score_panel.png')
    score_img = [pygame.image.load('./sources/menu/menu_score_0.png'),
                 pygame.image.load('./sources/menu/menu_score_1.png'),
                 pygame.image.load('./sources/menu/menu_score_2.png'),
                 pygame.image.load('./sources/menu/menu_score_3.png'),
                 pygame.image.load('./sources/menu/menu_score_4.png'),
                 pygame.image.load('./sources/menu/menu_score_5.png'),
                 pygame.image.load('./sources/menu/menu_score_6.png'),
                 pygame.image.load('./sources/menu/menu_score_7.png'),
                 pygame.image.load('./sources/menu/menu_score_8.png'),
                 pygame.image.load('./sources/menu/menu_score_9.png')]
    result_img = [pygame.image.load('./sources/gameover/result_score_0.png'),
                  pygame.image.load('./sources/gameover/result_score_1.png'),
                  pygame.image.load('./sources/gameover/result_score_2.png'),
                  pygame.image.load('./sources/gameover/result_score_3.png'),
                  pygame.image.load('./sources/gameover/result_score_4.png'),
                  pygame.image.load('./sources/gameover/result_score_5.png'),
                  pygame.image.load('./sources/gameover/result_score_6.png'),
                  pygame.image.load('./sources/gameover/result_score_7.png'),
                  pygame.image.load('./sources/gameover/result_score_8.png'),
                  pygame.image.load('./sources/gameover/result_score_9.png')]
    gold = pygame.image.load('./sources/gameover/medals_1.png')
    silver = pygame.image.load('./sources/gameover/medals_2.png')
    corper = pygame.image.load('./sources/gameover/medals_3.png')

    best = 0

    while True:
        Main()
